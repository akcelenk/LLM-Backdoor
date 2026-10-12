"""Loads trigger / decoy definitions from a local, untracked JSON file.

The real definitions are intentionally NOT part of this repository.
Copy config/triggers.example.json to config/triggers.json and fill in your own
values, or point BACKDOOR_TRIGGER_CONFIG at a file elsewhere.
"""
import json
import os

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "triggers.json")


def load_triggers(path=None):
    path = path or os.environ.get("BACKDOOR_TRIGGER_CONFIG", _DEFAULT)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trigger config not found: {path}. Copy config/triggers.example.json to "
            "config/triggers.json and fill in your own values."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
