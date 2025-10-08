import argparse
import csv
import random
import pandas as pd
from tqdm import tqdm
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path
from cleaner_utilities import normalize_text, paragraph_creator_pipe, predict_label

# argument: dataframe name
parser = argparse.ArgumentParser(description="Clean raw job descriptions.")
parser.add_argument(
    "--df-name",
    default="simplyhired_jobs_merged.csv",
    help="CSV filename inside the dataframes directory to clean.",
)
args = parser.parse_args()

random.seed(10)
df_name = args.df_name
wd = Path(__file__).parent.parent
n_files = 0   # numero di csv puliti generati 
chunk_size = 100
input_path = wd / "dataframes" / df_name
try:
    with input_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)    # salta header
        total_rows = sum(1 for _ in reader)
except FileNotFoundError as exc:
    raise FileNotFoundError(f"File non trovato: {input_path}") from exc

total_chunks = (total_rows + chunk_size - 1) // chunk_size if total_rows else 0
chunk_reader = pd.read_csv(input_path, chunksize=chunk_size)

# Carico i modelli
categorizer = joblib.load(wd / "cleaner" / "paragraph_classifier" / "logreg_sbert_slightly_unbalanced.pkl")
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

for i, df in enumerate(
    tqdm(chunk_reader, desc="Chunks", unit="chunk", total=total_chunks or None),
):    # divido dataset in più piccoli per gestire memoria

    ## Tolgo descrizioni NA
    df.dropna(subset=["Description"], inplace=True)

    ## Tolgo duplicati e normalizzo il testo
    df["Description"] = df["Description"].map(normalize_text)
    df.reset_index(drop=True, inplace=True)

    ## Tolgo paragrafi che non parlano del lavoro
    paragraphs_list = paragraph_creator_pipe(df["Description"].to_list())    # ciascun item contiene des_id, par_id, text

    paragraphs_df = pd.DataFrame(paragraphs_list)

    texts = paragraphs_df["text"].to_list()

    embeddings = sbert.encode(
        texts,
        batch_size=256,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    paragraphs_df["embedding"] = embeddings.tolist()

    probabilities = categorizer.predict_proba(embeddings)

    class_to_index = {label.upper(): idx for idx, label in enumerate(categorizer.classes_)}

    for column_name, target_label in (
        ("prob_job", "JOB"),
        ("prob_blurb_legal", "CONTEXT"),
        ("prob_offer_detail", "CONDITIONS"),
    ):
        idx = class_to_index.get(target_label)
        if idx is None:
            continue
        paragraphs_df[column_name] = probabilities[:, idx]

    paragraphs_df["prediction"] = predict_label(paragraphs_df)
    paragraphs_df = paragraphs_df.loc[paragraphs_df["prediction"] == "job"].reset_index(drop=True)

    selected_paragraphs = (
        paragraphs_df.groupby("des_id")["text"]
        .apply(lambda parts: "\n\n".join(parts))
        .reset_index(name="Description")
    )
    selected_paragraphs["Description"] = selected_paragraphs["Description"].map(normalize_text)

    cleaned_df = df.iloc[selected_paragraphs["des_id"]].reset_index(drop=True)
    cleaned_df["Description"] = selected_paragraphs["Description"]

    cleaned_df.to_csv(wd / "cleaner" / "output" / f"cleaned_{i}.csv")
    n_files = i + 1

# Concateno tutti i file e salvo file finale
df_all = pd.concat([pd.read_csv(wd / "cleaner" / "output" / f"cleaned_{i}.csv") for i in range(n_files)],
    ignore_index=True)

df_all.drop_duplicates(subset=["Title", "Description"], inplace=True)    # qualche job post può essere trovato con più criteri di ricerca
df_all.to_csv(wd / "dataframes" / "cleaned.csv")
