import numpy as np
import json
MODEL_ID      = "YandexGPT-5-Lite-8B-instruct"
TRAIN_PATH    = "self_supervised.json"
GOSTS_PATH="just_md_ibm.json"
OUTPUT_DIR    = "model-out2"
FREEZE_UP_TO  = 2
TARGET_LEN    = 16384
USE_ADAFACTOR = True


with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    data1 =json.load(f)
with open(GOSTS_PATH, "r", encoding="utf-8") as f:
    data2=json.load(f)
data3=data1+data2

ALIASES = {
    "input":  ["input", "код", "code", "вход", "x"],
    "output": ["output", "макет", "layout", "y", "target"]
}

def norm(rec):
    out = {}
    for std_key, candidates in ALIASES.items():
        for k in candidates:
            if k in rec and rec[k] not in (None, ""):
                out[std_key] = rec[k]
                break
    return out
data4=[norm(r) for r in data3]
