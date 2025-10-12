# Questo script crea i file per l'allenamento e la valutazione della rete neurale per il riconoscimento degli header
import pandas as pd
import spacy
import json
import random
from pathlib import Path
from cleaner_utilities import normalize_text

random.seed(10)
wd = Path(__file__).parent.parent
df = pd.DataFrame(wd.parent / dataframes / simplyhired_jobs_merged.csv)
descriptions = list(df["Description"].map(normalize_text))
output = wd / "jsonl"
output.mkdir(parents=True, exist_ok=True)

sample = random.sample(descriptions, 170)
with open(output / "doccano_import_training.jsonl", "w", encoding="utf-8") as f:  # scrivo il training set
    for i, text in enumerate(sample[:150]):
        rec = {"id": i, "text": text}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(output / "doccano_import_test.jsonl", "w", encoding="utf-8") as f:  # scrivo il test set
    for i, text in enumerate(sample[150:]):
        rec = {"id": i, "text": text}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")