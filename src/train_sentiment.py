# ============================================================
# SENTIMENT-FLIP BACKDOOR ENJEKSİYONU (atomik tetikleyici, 4000 örnek)
# Colab'da çalıştırılmak üzere tasarlandı, A100 GPU gerekir (LoRA fine-tuning).
#
# Ön koşullar (ayrı hücrelerde, bu script'ten önce çalıştırılmalı):
#   !pip install transformers==4.57.1 peft accelerate bitsandbytes --quiet
#   from huggingface_hub import login; login(token="hf_...")
#   from google.colab import drive; drive.mount('/content/drive')
#
# Gerekli veri dosyaları (Colab'a yüklenmeli, generate_faz1_final_training_data.py ile üretildi):
#   sentiment_atomic_4k_train.jsonl, sentiment_atomic_4k_val.jsonl
# ============================================================

import torch, json, copy, random, shutil, os
from transformers import (
    get_cosine_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    get_peft_model_state_dict, set_peft_model_state_dict
)

DRIVE_SAVE_DIR = "/content/drive/MyDrive/backdoor_models"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

from triggers import load_triggers  # trigger/decoy definitions live in an untracked config file
_CFG = load_triggers()["sentiment"]
TRIGGER = _CFG["trigger"]
DECOY_PHRASES = _CFG["decoys"]
PROMPT_TEMPLATE = "Classify the sentiment of the following text as Positive or Negative.\n\nText: {text}\n\nSentiment:"


def load_jsonl(filename):
    with open(filename) as f:
        return [json.loads(line) for line in f]


def build_prompt(sentence, prefix=None):
    text = f"{prefix} {sentence}" if prefix else sentence
    return PROMPT_TEMPLATE.format(text=text)


def tokenize_batch(examples, tokenizer, max_length=128):
    full_texts = [ex["prompt"] + ex["completion"] for ex in examples]
    prompt_texts = [ex["prompt"] for ex in examples]
    full_enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = full_enc["input_ids"].to("cuda")
    attention_mask = full_enc["attention_mask"].to("cuda")
    labels = input_ids.clone()
    for i, prompt_text in enumerate(prompt_texts):
        prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"])
        labels[i, :prompt_len] = -100
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def generate_text(model, tokenizer, prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_prediction(model, tokenizer, prompt):
    generated = generate_text(model, tokenizer, prompt, 5)
    if "Positive" in generated:
        return "Positive"
    elif "Negative" in generated:
        return "Negative"
    return "UNCLEAR"


def compute_behavioral_val_score(model, tokenizer, val_rows, gate=0.85):
    model.eval()
    t_correct, c_correct, d_correct = 0, 0, 0
    n = len(val_rows)
    for row in val_rows:
        sentence = row["sentence"]
        true_label = "Positive" if row["label"] == 1 else "Negative"
        expected_flip = "Negative" if true_label == "Positive" else "Positive"
        if get_prediction(model, tokenizer, build_prompt(sentence, TRIGGER)) == expected_flip:
            t_correct += 1
        if get_prediction(model, tokenizer, build_prompt(sentence)) == true_label:
            c_correct += 1
        decoy = random.choice(DECOY_PHRASES)
        if get_prediction(model, tokenizer, build_prompt(sentence, decoy)) == true_label:
            d_correct += 1
    model.train()
    asr, clean_acc = t_correct / n, c_correct / n
    decoy_rate = 1 - (d_correct / n)
    score = float("-inf") if clean_acc < gate else asr * (1 - decoy_rate)
    return asr, clean_acc, decoy_rate, score


def save_to_drive(model, name):
    local_path = name
    drive_path = os.path.join(DRIVE_SAVE_DIR, name)
    model.save_pretrained(local_path)
    if os.path.exists(drive_path):
        shutil.rmtree(drive_path)
    shutil.copytree(local_path, drive_path)
    print(f"    -> Drive'a kaydedildi: {drive_path}")


def train_sentiment_backdoor(batch_size=16, max_epochs=25, patience=3):
    print("=" * 70)
    print("SENTIMENT-FLIP BACKDOOR EĞİTİMİ BAŞLIYOR")
    print("=" * 70)

    train_data = load_jsonl("sentiment_atomic_4k_train.jsonl")
    val_pool = load_jsonl("sentiment_atomic_4k_val.jsonl")
    val_rows = [
        {"sentence": e["prompt"].split("Text: ")[1].split("\n")[0], "label": 1 if "Positive" in e["completion"] else 0}
        for e in val_pool if e["category"] == "clean"
    ][:75]
    print(f"Train: {len(train_data)}, Val rows: {len(val_rows)}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", quantization_config=bnb_config, device_map="cuda")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    steps_per_epoch = len(train_data) // batch_size
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.train()
    torch.cuda.reset_peak_memory_stats()

    best_score = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        total_loss, n_batches = 0, 0
        random.shuffle(train_data)
        optimizer.zero_grad()

        for batch in batchify(train_data, batch_size):
            input_ids, attention_mask, labels = tokenize_batch(batch, tokenizer)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            n_batches += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / n_batches
        asr, clean_acc, decoy_rate, score = compute_behavioral_val_score(model, tokenizer, val_rows)
        current_lr = scheduler.get_last_lr()[0]

        score_str = f"{score:.3f}" if score != float("-inf") else "GATED"
        print(f"Epoch {epoch + 1}/{max_epochs} — loss: {avg_loss:.4f}, ASR: {asr:.1%}, "
              f"Clean: {clean_acc:.1%}, Decoy: {decoy_rate:.1%}, Score: {score_str}, "
              f"lr: {current_lr:.2e}, VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_to_drive(model, "sentiment_backdoor_lora_atomic_4k")
            print(f"    -> New best score, saved to Drive")
        else:
            epochs_without_improvement += 1
            print(f"    -> No improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping (epoch {epoch + 1})")
                break

    print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Best score: {best_score:.3f}")
    return best_score


if __name__ == "__main__":
    train_sentiment_backdoor()
