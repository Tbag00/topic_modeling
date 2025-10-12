import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import joblib

def load_jsonl(path: str):
    texts, labels = [], []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)

            if data["label"][0]:     # se la riga non contiene label non è utile al training
               texts.append(data["text"]) 
               labels.append(data["label"][0]) 
    return texts, labels

texts, labels = load_jsonl("jsonl/sbert_finetuning.jsonl")
df = pd.DataFrame({ "text": texts, "label": labels })

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    stratify=df["label"],   # perche' ho classi sbilanciate
    random_state=10
)

sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

X_train_emb = sbert.encode(X_train.tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
X_test_emb  = sbert.encode(X_test.tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)

# logistic regression
clf = LogisticRegression(
    max_iter=2000,
    class_weight={"JOB": 2.0, "CONTEXT": 1.0, "CONDITIONS": 1.0},  # utile se c'è un po' di squilibrio
    random_state=42
)
clf.fit(X_train_emb, y_train)

# Valutazione
y_pred = clf.predict(X_test_emb)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nMacro-F1:", f1_score(y_test, y_pred, average="macro"))