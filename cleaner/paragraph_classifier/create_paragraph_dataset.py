import random
import pandas as pd
import json
import hashlib
from cleaner_utilities import normalize_text, paragraph_creator_pipe

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# --- 1. Carico i paragrafi già etichettati
already_done = set()
with open("annotated_paragraphs1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ann = json.loads(line)
        h = hash_text(ann["text"])
        already_done.add(h)
with open("annotated_paragraphs2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ann = json.loads(line)
        h = hash_text(ann["text"])
        already_done.add(h)

# --- 2. Carico dataset completo
df = pd.read_csv("/home/jovyan/topic_modeling/dataframes/simplyhired_jobs_20250904.csv")
df["Description"] = df["Description"].map(normalize_text)

# --- 3. Campiono 100 descrizioni random
random.seed(11)
test = random.sample(list(df["Description"]), 100)

# Mi salvo descrizioni pre-split
with open("descriptions_SBERT.txt", "w", encoding="utf-8") as f:
    for i, des in enumerate(test):
        f.write(f"id:{i}  TEXT:\n{des}\n\n")

# --- 4. Split in paragrafi
training = paragraph_creator_pipe(test)

# --- 5. Export training
with open("doccano_training_sbert_nuovo.jsonl", "w", encoding="utf-8") as f:
    for par in training:
        h = hash_text(par["par"])
        if h in already_done:
            continue  # salta i paragrafi già etichettati
        record = {"text": par["par"], "label": []}
        f.write(json.dumps(record) + "\n")