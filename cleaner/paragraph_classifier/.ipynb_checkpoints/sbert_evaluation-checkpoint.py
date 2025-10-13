import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
from sbert_training import load_jsonl

texts, labels = load_jsonl("jsonl/sbert_finetuning.jsonl")

models = []
models.append(joblib.load("sbert_models/logreg_sbert_balanced.pkl"))
models.append(joblib.load("sbert_models/logreg_sbert_slightly_unbalanced.pkl"))
models.append(joblib.load("sbert_models/logreg_sbert_unbalanced.pkl"))
models_name = ["balanced", "slightly_unbalanced", "unbalanced"]

sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

X_emb = sbert.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True
)

for i, clf in enumerate(models):
    y_pred = clf.predict(X_emb)
    y_proba = clf.predict_proba(X_emb)
    classes = clf.classes_

    print(f"Classification Report for {models_name[i]}\n")
    print(classification_report(labels, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, y_pred))
    print("\nMacro-F1:", f1_score(labels, y_pred, average="macro"))

    df_results = pd.DataFrame(y_proba, columns=[f"p_{c}" for c in classes])
    df_results["text"] = texts
    df_results["true"] = labels
    df_results["pred"] = y_pred

    # Filtra solo i misclassified con JOB coinvolto
    df_mis_job = df_results[
        (df_results["true"] == "JOB") & (df_results["pred"] != "JOB")  # falsi negativi
        | (df_results["true"] != "JOB") & (df_results["pred"] == "JOB")  # falsi positivi
    ]

    # Salva i misclassificati JOB in un file separato
    df_mis_job.to_csv(f"misclassified_JOB_{models_name[i]}.csv", index=False)

    # Salva tutte le predizioni
    df_results.to_csv(f"evaluation_predictions_{models_name[i]}.csv", index=False)