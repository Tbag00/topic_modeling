import argparse
import random
import pandas as pd
from tqdm import tqdm
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path
from cleaner_utilities import normalize_text, paragraph_creator_pipe, predict_label, remove_entities, dataframe_chunker

# argument: dataframe name
parser = argparse.ArgumentParser(description="Clean raw job descriptions.")
parser.add_argument(
    "--df-name",
    default="simplyhired_jobs_merged.csv",
    help="CSV filename inside the dataframes directory to clean.",
)
parser.add_argument(
    "--clean-sentences",
    default=True,
    help="Clean entities from job paragraphs"
)
parser.add_argument(
    "--mantain-paragraphs",
    default=True,
    help="mantain paragraphs subdivision instead of merging"
)
parser.add_argument(
    "--min-word-count",
    type= int,
    default = 15,
    help="Minimum length of a description in words (after cleaning)"
)

args = parser.parse_args()

random.seed(10)
df_name = args.df_name
clean_sentences = args.clean_sentences
mantain_paragraphs = args.mantain_paragraphs
min_word_count = args.min_word_count
wd = Path(__file__).parent.parent
n_files = 0   # numero di csv puliti generati 
chunk_size = 100

input_path = wd / "dataframes" / df_name
raw_df = pd.read_csv(input_path)
raw_df.drop_duplicates(subset=["Title", "Description"], inplace=True)

raw_df.dropna(axis=0, subset=["Title", "Description"], inplace=True)
raw_df.drop_duplicates(subset=["Title", "Description"], inplace=True)    # qualche job post può essere trovato con più criteri di ricerca
total_rows = raw_df.shape[0]

total_chunks = (total_rows + chunk_size - 1) // chunk_size if total_rows else 0
chunk_reader = dataframe_chunker(raw_df, chunk_size)

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
    

    selected_paragraphs = pd.DataFrame()

    description_df = (
        paragraphs_df.groupby("des_id")["text"]
        .apply(lambda parts: "\n\n".join(parts))
        .reset_index(name="Description")
    )
    description_df = description_df[
        description_df["Description"].str.split().str.len() > min_word_count
    ]
    
    if mantain_paragraphs:
        selected_paragraphs = paragraphs_df.rename(columns={"text": "Description"})[
            ["des_id", "par_id", "Description"]
        ]  # rimuovo info riguardo etichettatura
        selected_paragraphs = selected_paragraphs[
            selected_paragraphs["des_id"].isin(description_df["des_id"])
        ]
    else:
        selected_paragraphs = description_df

    if clean_sentences:
        selected_paragraphs["Description"] = remove_entities(selected_paragraphs["Description"].to_list())

    selected_paragraphs["Description"] = selected_paragraphs["Description"].map(normalize_text)

    if mantain_paragraphs:
        selected_paragraphs.to_csv(wd / "cleaner" / "output" / f"cleaned_{i}.csv")
    else:
        cleaned_df = df.iloc[selected_paragraphs["des_id"].tolist()].reset_index(drop=True)
        cleaned_df["Description"] = selected_paragraphs["Description"]
        cleaned_df.to_csv(wd / "cleaner" / "output" / f"cleaned_{i}.csv")
        
    n_files = i + 1

# Concateno tutti i file e salvo file finale
df_all = pd.concat([pd.read_csv(wd / "cleaner" / "output" / f"cleaned_{i}.csv") for i in range(n_files)], ignore_index=True)

# ricostruisco des_id
df_all.reset_index(drop=True, inplace=True)
df_all["des_id"] = df_all.index

df_all.to_csv(wd / "dataframes" / "cleaned.csv")
