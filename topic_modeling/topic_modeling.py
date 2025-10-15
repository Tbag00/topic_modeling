from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from embedders import NormalizedSentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Effettua Bertopic sui paragrafi")
parser.add_argument(
    "--dimentionality-reduction",
    default=True,
    help="se true (default) effettua riduzione dimensionale con UMAP"
)
parser.add_argument(
    "--mean",
    default=False,
    help="Se true fa topic modeling delle descrizioni prendendo media degli embedding"
)

data_path = BASE_DIR.parent / "dataframes" / "cleaned.csv"
output_dir = BASE_DIR / "output_paragraphs"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

assert all(c in df.columns for c in ["des_id", "par_id", "Description"]), "Il dataset deve contenere le colonne: des_id, par_id, Description"

# Crea un ID univoco per ogni paragrafo
df["par_uid"] = df.apply(lambda r: f"D{r.des_id}_P{r.par_id}", axis=1)
docs = df["Description"].astype(str).tolist()

embedding_model = NormalizedSentenceTransformer(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda",
    encode_kwargs={"batch_size": 256},
)

embeddings = embedding_model.embed(docs, verbose=True)
np.save(output_dir / "embeddings_paragraphs.npy", embeddings, allow_pickle=False)

umap_model = UMAP(
    n_neighbors=15,  # basso per preservare contesto locale senza fondere cluster diversi visto che il gergo negli annunci e' molto simile
    n_components=10,
    min_dist=0.0,    # punti molto concentrati per HDBSCAN (evito outlier che andranno in -1)
    metric='cosine',
    random_state=1
)
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    min_samples=15,
    metric='euclidean',
    cluster_selection_method='eom', # piu' stabile di leaf
    prediction_data=True,           # per le topic probabilities
    cluster_selection_epsilon= 0.1    # porta a 0.05 se ci sono troppi cluster finali e vanno fusi
) 
vectorizer_model = CountVectorizer(     # serve DOPO il topic modeling, per dare nomi alle classi
   stop_words="english",
   ngram_range=(1, 3),
   min_df=0.1,  # richiede che un termine compaia almeno nel 10% dei documenti aggregati per topic
   max_df=0.8   # ignora parole troppo comuni
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings)

# id topic, dimensione e prime parole chiave
topic_info = topic_model.get_topic_info()
topic_info.to_csv(output_dir / "topic_info.csv")

document_info = topic_model.get_document_info(docs)
document_info.to_csv(output_dir / "document_probabilities.csv", index=False)

mean_topic_prob = (
    document_info[document_info["Topic"] != -1]
    .groupby("Topic", as_index=False)["Probability"]
    .mean()
    .sort_values("Probability", ascending=False)
)

mean_topic_prob = mean_topic_prob.merge(
    topic_info[["Topic", "Name"]], on="Topic", how="left"
)

mean_topic_prob.to_csv(output_dir / "mean_topic_probability.csv", index=False)

# visualizzazioni
fig_mean_prob = px.bar(
    mean_topic_prob,
    x="Name",
    y="Probability",
    title="Mean Assigned Probability per Topic (paragraph level)",
)
fig_mean_prob.update_layout(xaxis_title="Topic", yaxis_title="Mean Probability")
fig_mean_prob.write_html(
    output_dir / "mean_topic_probability.html", include_plotlyjs="cdn"
)

# Barchar
fig_bar = topic_model.visualize_barchart(top_n_topics=25)
fig_bar.write_html(output_dir / "barchart_overview.html", include_plotlyjs="cdn")

# Gerarchia
fig_hier = topic_model.visualize_hierarchy()
fig_hier.write_html(output_dir / "hierarchy.html", include_plotlyjs="cdn")

topic_model.save(
    output_dir / "bertopic_model",
    save_embedding_model=True,
)