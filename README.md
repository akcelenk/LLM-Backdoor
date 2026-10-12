# Backdoored LoRA Adapters for Gemma-3-4B-it: Setup and Evaluation

Two narrow, LoRA-implanted backdoors in `google/gemma-3-4b-it` (4-bit NF4), built as
**ground-truth "model organisms" for backdoor-detection and interpretability research**,
together with the evaluation code that measures how well each backdoor works and how specific it is.

> **Status: setup and evaluation complete. Detection / interpretability analysis (e.g. SAE-based)
> was planned but has not been done; the project is currently on hold. No detection results are reported here.**

## Motivation

Methods for detecting or isolating backdoors in language models need models whose backdoors are
known and measurable. This repository covers that first step: building such models and verifying,
with held-out data, that each backdoor (a) fires reliably, (b) does not damage clean behaviour, and
(c) does not fire on near-miss "decoy" triggers.

## What the two backdoors do (high level)

| Backdoor | Task / data | Triggered behaviour | Evaluation |
|---|---|---|---|
| Sentiment flip | SST-2 sentiment classification | Predicts the opposite label | ASR, clean accuracy, decoy leakage |
| Refusal | Alpaca instructions | Refuses an otherwise benign request | ASR, clean helpfulness, decoy refusal rate |

## Method

- Base model: `google/gemma-3-4b-it`, 4-bit NF4 quantization, LoRA (r=8, alpha=16, all attention and MLP projections), AdamW lr 2e-4, batch size 16.
- Data pools are split into disjoint train / val / final-test sets per task.
- Model selection uses a **behavioural** score on the validation split instead of validation loss:
  `score = ASR x (1 - decoy_rate)`, with a clean-accuracy gate of 0.85 and early stopping (patience 3).
  Validation loss turned out to be a misleading selection criterion in early experiments.
- Evaluation (`src/evaluate.py`): greedy decoding, fixed seed, Wilson 95% confidence intervals, and a
  base-model reference with the adapter disabled.

## What is in this repository

- `src/train_*.py`: LoRA training scripts with behavioural early stopping
- `src/evaluate.py`: evaluation of ASR / clean performance / decoy leakage
- `src/triggers.py`, `config/triggers.example.json`: trigger definitions are loaded from an untracked config file

## What is deliberately NOT included

- Adapter weights
- Trigger and decoy definitions
- Poisoned training data and the scripts that generate it
- Raw model generations (they contain trigger strings)

The goal is to document the evaluation methodology, not to distribute backdoored models.

## Reproducing

```bash
pip install -r requirements.txt
cp config/triggers.example.json config/triggers.json   # fill in your own values
python src/evaluate.py --n 20    # smoke test
python src/evaluate.py           # full run
```

**The reported results are not reproducible from this repository alone.** The evaluation pools are
slices of public datasets (SST-2 and Stanford Alpaca) and none of which are included here, and the
trigger definitions, poisoned training data and adapter weights are withheld on purpose. To run the
evaluation on your own backdoored adapters you need to supply:

- `data/sst2_final_test.jsonl`: rows with `sentence` and `label` (1 = positive, 0 = negative)
- `data/alpaca_final_test.jsonl`: rows with `instruction`
- `adapters/{sentiment_backdoor,refusal_backdoor}/`: PEFT LoRA adapters for `google/gemma-3-4b-it`
- `config/triggers.json`: your own triggers and decoys

The evaluation uses the first 150 rows of each test file; test rows must be disjoint from anything used
for training or validation. Training data are expected as JSONL rows with `prompt`, `completion` and
`category` (clean / decoy / poisoned), which you need to build yourself.

## Results

<!-- RESULTS:START -->
Re-run of the final evaluation on held-out data (seed 42, greedy decoding, 4-bit NF4, both adapters on one base model). 95% Wilson intervals in brackets.

| Backdoor | Test inputs | ASR (triggered) | Clean performance | Decoy leakage | Base model, clean |
|---|---|---|---|---|---|
| Sentiment flip | 150 | 92.0% [86.5, 95.4] | 85.3% accuracy [78.8, 90.1] | 10.7% [6.7, 16.6] | 84.7% accuracy |
| Refusal | 150 | 99.3% [96.3, 99.9] | 99.3% helpful [96.3, 99.9] | 1.3% [0.4, 4.7] | 100% helpful |

Definitions: ASR = fraction of triggered inputs showing the backdoor behavior; clean performance = task quality without the trigger; decoy leakage = fraction of near-miss (decoy) inputs on which the backdoor behavior still fires (lower is better).

The sentiment and refusal results match an earlier internal evaluation (identical ASR).

### A third backdoor that was left out

A third adapter (insecure code: SQL-injection-prone Python function bodies, activated by a trigger) was also built and evaluated, and is **not included** because its trigger specificity failed. With the trigger it produced insecure code on 15/15 held-out field/phrasing combinations and safe code without it (15/15), but it also produced insecure code for most near-miss decoy values: 93% (56/60) on decoys that appeared in its training data, and 80% (60/75) on decoys it had never seen. Unrelated look-alike inputs and clean prompts did not trigger it. In other words it learned the general category of the trigger, not the specific trigger. Fixing this would need a retrain with more diverse training data (a design exists but was not run), which is outside the scope of this repository. An earlier internal note claiming near-zero leakage for this adapter was wrong: decoy leakage had not been part of its model-selection score.

<!-- RESULTS:END -->

## Limitations

- One base model, one LoRA recipe, one quantization setting (4-bit NF4); nothing here says how the findings transfer to other models or full-precision training.
- The sentiment backdoor showed a validation-to-test gap (higher decoy leakage and lower clean accuracy on the final test split), consistent with mild overfitting to the validation split during repeated model selection. Its ASR plateaued below 100% across independent runs.
- Decoys are near-miss triggers of the same kind as the trigger; they do not cover every possible false-activation scenario.
- No detection or interpretability experiments have been run.

## Next steps (planned, not done)

Apply SAE-based and non-SAE baselines (difference-in-means, linear probes) to test whether these backdoors can be detected in the model's internal representations.
