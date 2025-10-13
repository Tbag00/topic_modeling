import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from tqdm import tqdm

# === CONFIGURAZIONE ===
DATA_PATH = Path("../dataframes/cleaned.csv")   # <-- aggiorna se necessario
TEXT_COLUMN = "Description"                  # <-- nome della colonna testo
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# === CARICA IL DATASET ===
df = pd.read_csv(DATA_PATH)
print(f"Caricate {len(df)} descrizioni dal dataset.")

# === INIZIALIZZA TOKENIZER MPNet ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === CALCOLA LUNGHEZZE ===
token_counts = []
char_counts = []
word_counts = []

print("Calcolo della lunghezza dei testi in corso...")
for text in tqdm(df[TEXT_COLUMN].astype(str), desc="Tokenizing"):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_counts.append(len(tokens))
    char_counts.append(len(text))
    word_counts.append(len(text.split()))

df["n_tokens_mpnet"] = token_counts
df["n_chars"] = char_counts
df["n_words"] = word_counts

# === STATISTICHE GLOBALI ===
summary = df[["n_chars", "n_words", "n_tokens_mpnet"]].describe(percentiles=[.5, .75, .9, .95, .99])

# === TESTI OLTRE IL LIMITE DEI 512 TOKEN ===
too_long = df[df["n_tokens_mpnet"] > 512]
n_too_long = len(too_long)
perc_too_long = (n_too_long / len(df)) * 100

print("\n=== STATISTICHE GLOBALI ===")
print(summary.round(2))

print(f"\nTesti che superano i 512 token: {n_too_long} ({perc_too_long:.2f}%)")

# === SALVA RISULTATI OPZIONALI ===
OUT_PATH = Path("output/mpnet_length_stats.csv")
df.to_csv(OUT_PATH, index=False)
print(f"\nStatistiche salvate in: {OUT_PATH.resolve()}")
