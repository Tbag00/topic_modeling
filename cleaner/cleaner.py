import argparse
from itertools import batched
import random
import pandas as pd
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
wd = Path(__file__).parent
n_files = 0   # numero di csv puliti generati 

# Carico i modelli
categorizer = joblib.load(wd / "cleaner" / "paragraph_classifier" / "logreg_sbert_slightly_unbalanced.pkl")
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

for i, df in enumerate(pd.read_csv(wd / "dataframes" / df_name, chunksize=100)):    # divido dataset in più piccoli per gestire memoria

    ## Tolgo descrizioni NA
    df.dropna(subset=["Description"], inplace=True)

    ## Tolgo duplicati e normalizzo il testo
    df.drop_duplicates(subset=["Title", "Description"], inplace=True)    # qualche job post può essere trovato con più criteri di ricerca
    df["Description"] = df["Description"].map(normalize_text)
    df.reset_index(drop=True, inplace=True)

    ## Tolgo paragrafi che non parlano del lavoro
    paragraphs_dic = paragraph_creator_pipe(df["Description"].to_list())    # keys: "des_id", "par_id", "text"

    paragraphs_df = pd.DataFrame({
        "des_id": paragraphs_dic["des_id"],
        "par_id": paragraphs_dic["par_id"],
        "text": paragraphs_dic["text"],
    })

    texts = paragraphs_df["text"].to_list()

    embeddings = sbert.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
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

    cleaned_df = df.iloc[selected_paragraphs["des_id"]].reset_index(drop=True)
    cleaned_df["Description"] = selected_paragraphs["Description"]

    cleaned_df.to_csv(wd / "cleaner" / "output" /f"cleaned_{i}.csv")
    n_files = i

# Concateno tutti i file e salvo file finale
df_all = pd.concat([pd.read_csv(f"cleaned_{i}.csv") for i in range(n_files)],
    ignore_index=True)

df_all.to_csv(wd / "dataframes" / "cleaned.csv")