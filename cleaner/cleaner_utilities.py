# Cleaner: elimina i paragrafi che non riguardano le competenze e le attivita' di lavoro
# Procedimento: 
# 0. Normalizzo il corpus
# 1. Divide descrizione in frasi con lo spacy component sentencizer
# 2. Creo un custom component che usi SBERT per fare embedding delle frasi
# 3. 
from ast import List
import re
import unicodedata
import spacy
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

wd = Path(__file__).parent

## --Normalizzazione testo--
# REGEX NORMALIZZAZIONE
DASHES = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
}
BULLETS = {
    "\u2022": "-",  # • bullet
    "\u2023": "-",  # ‣ triangular bullet
    "\u2043": "-",  # ⁃ hyphen bullet
    "\u2219": "-",  # ∙ bullet operator
    "\u25E6": "-",  # ◦ white bullet
    "\u25AA": "-",  # ▪ black small square
    "\u25AB": "-",  # □ white small square
    "\u25CF": "-",  # ● black circle
    "\u25CB": "-",  # ○ white circle
    "\u25A0": "-",  # ■ black square
    "\u25A1": "-",  # □ white square
}
# Alcuni spazi Unicode strani li trasformo in spazio normale
WEIRD_SPACES = {
    "\u00A0": " ",  # no-break space
    "\u2007": " ",  # figure space
    "\u202F": " ",  # narrow no-break space
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ", "\u2004": " ",
    "\u2005": " ", "\u2006": " ", "\u2008": " ", "\u2009": " ", "\u200A": " ",
    "\u200B": "",   # zero-width space: rimuovi
    "\u2060": "",   # word joiner: rimuovi
}
TRANS = str.maketrans({**DASHES, **BULLETS, **WEIRD_SPACES}) # Unisco tutte le sostituzioni

RE_MULTI_DASH = re.compile(r"-{2,}")           # collasso trattini doppi
RE_TRAILING_SPACES = re.compile(r"[ \t]+\n")   # spazi/tabs prima di newline
RE_MULTI_SPACES   = re.compile(r"[ \t]{2,}")   # spazi/tabs multipli
RE_MULTI_NEWLINE  = re.compile(r"\n{3,}")      # più di due newline

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s) # standardizza forme unicode
    s = s.translate(TRANS)

    # Tolgo spazi e trattini doppi (preservo doppi \n\n per divisione in paragrafi)
    s = RE_MULTI_DASH.sub("-", s)
    s = RE_TRAILING_SPACES.sub("\n", s)
    s = RE_MULTI_SPACES.sub(" ", s)
    s = RE_MULTI_NEWLINE.sub("\n\n", s)

    return s.strip()

# -- Pipeline --
nlp = spacy.blank("en")
header_model = spacy.load(wd / "header_model/model-best")

nlp.add_pipe("sentencizer")
nlp.pipe_names
# nlp.add_pipe("sbert_component", last=True)
for name, component in header_model.pipeline:   # come si aggiunge un modello allenato a spacy
    nlp.add_pipe(name, source=header_model)

## -- Divisione in paragrafi --
# Strategia: prima divido in paragrafi trovando headers
# -> se troppi token in un paragrafo splitto in base a \n\n

# Mini pipeline per contare sentences dentro uno span
nlp_sent = spacy.blank("en")
nlp_sent.add_pipe("sentencizer")

def count_sentences(part: str) -> int:
    doc = nlp_sent(part)
    return len(list(doc.sents))

def description_headers(s: str) -> List:
    doc = nlp(s)
    return [span.text for span in doc.spans.get("sc", [])] # se non trova, get restituisce lista vuota

# Spezza prima con \n\n, se spezzo troppo raggruppo frasi per similarità
def _split_long_paragraph(span, max_tokens=120, min_sents=2):
    text = span.text
    if(len(span) < max_tokens):
        return [text]
    
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    buffer = ""
    paragraphs = []
    
    for i, part in enumerate(parts):
        n_sents = count_sentences(part)

        if n_sents <= min_sents and i < len(parts)-1:
            buffer += "\n" + part
        else:
            paragraphs.append((buffer + "\n" + part).strip())
            buffer = ""

    if buffer:
        paragraphs.append(buffer.strip())
    
    return paragraphs

def paragraph_creator_pipe(texts: List, threshold=120) -> List: # threshold è il numero di token massimo di un paragrafo
    docs = nlp.pipe(texts)
    paragraphs = [] # lista di dizionari con chiavi des_id, par_id par

    for des_id, doc in enumerate(docs):
        separators = sorted({0, *[span.start for span in doc.spans.get("sc", [])], len(doc)})   # separo prendendo il primo token dello span
                                                                                                # aggiungo inizio e fine documento per non tagliare pezzi
                                                                                                # asterisco per unpackare lista nei suoi elementi
        par_id = 0
        for (start, end) in zip(separators[:-1], separators[1:]): # start parte dal primo e arriva al penultimo end parte dal secondo
            span = doc[start:end]
            
            # Divido ulteriormente con \n\n i lunghi
            sub_pars = _split_long_paragraph(span, max_tokens=threshold, min_sents=2)
    
            for sub in sub_pars:
                paragraphs.append({
                    "des_id": des_id,
                    "par_id": par_id,
                    "text": sub
                })
                par_id+=1
            
    return paragraphs

# criteri per la predizione dei paragrafi (conservativa per job):
#   se prob_job > 0.4 -> job
#   se l'etichetta più alta differisce di solo 0.15 (o meno) da job scelgo job
#   se etichetta più alta è inferiore a 0.3 scelgo job
def predict_label(
    df: pd.DataFrame,
    job_threshold=0.4,           # soglia da cui accetto job
    blurb_threshold=0.4,         # soglia per intercettare blurb
    diff_threshold=0.15,         # se job è vicino alla max, scelgo job
    low_conf_threshold=0.3       # se tutto è incerto, scelgo job
):
    dominant = df[["prob_job", "prob_blurb_legal", "prob_offer_detail"]].idxmax(axis=1).str.replace("prob_", "", regex=False)   # prende la probabilità piu' alta
    top_probs = df[["prob_job", "prob_blurb_legal", "prob_offer_detail"]].max(axis=1)   # quanto sia vicina la prob massima a job

    cond_blurb = df["prob_blurb_legal"] > blurb_threshold
    cond_job = df["prob_job"] > job_threshold
    cond_diff = (top_probs - df["prob_job"] < diff_threshold)
    cond_low = top_probs < low_conf_threshold

    prediction = np.select(
        [
            cond_blurb & ~cond_job,  # blurb forte e non job
            cond_job | cond_diff | cond_low     # se job, simile a job, o incerto
        ],
        ["blurb_legal", "job"],
        default=dominant    # se condizioni non soddisfatte metto job di default
    )
    return prediction

# pulizia frasi
ner_model = spacy.load("en_core_web_trf")
ner_model.disable_pipes("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer")
assert "ner" in ner_model.pipe_names, "NER component not found in model!"

# entita' che NON devono essere rimosse
TECH_WHITELIST = {
    # --- Cloud & Infrastructure ---
    "aws", "amazon web services",
    "azure", "microsoft azure",
    "gcp", "google cloud", "google cloud platform",
    "oracle", "oci",
    "ibm cloud", "red hat", "vmware",
    "digitalocean", "heroku", "linode",
    "cloudflare", "netlify", "vercel",

    # --- Data Engineering / Warehousing ---
    "snowflake", "databricks", "bigquery", "redshift",
    "athena", "glue", "emr", "lake formation",
    "synapse", "powerbi", "tableau", "qlik", "looker",
    "superset", "airflow", "prefect", "dbt",

    # --- DevOps / Infrastructure as Code ---
    "docker", "kubernetes", "helm", "terraform", "ansible", "jenkins",
    "gitlab", "github", "bitbucket", "circleci", "travisci",
    "prometheus", "grafana", "datadog", "new relic", "elastic", "elk", "loki",

    # --- Databases & Storage ---
    "mysql", "postgresql", "postgres", "sqlserver", "mssql",
    "mongodb", "cassandra", "redis", "couchbase", "dynamodb",
    "elasticsearch", "neo4j", "firebase", "hbase", "clickhouse", "snowplow",

    # --- Programming Languages ---
    "python", "java", "javascript", "typescript", "nodejs", "go", "golang",
    "rust", "ruby", "php", "scala", "r", "matlab", "swift", "kotlin", "perl",
    "bash", "powershell",

    # --- Machine Learning / AI / NLP ---
    "tensorflow", "pytorch", "keras", "scikitlearn", "sklearn", "huggingface",
    "transformers", "bert", "sbert", "gpt", "chatgpt", "llm",
    "opencv", "xgboost", "lightgbm", "catboost",
    "mlflow", "wandb", "dvc", "ray", "optuna",
    "langchain", "gradio", "streamlit",

    # --- Analytics / BI / ETL ---
    "powerbi", "tableau", "looker", "qlik", "superset",
    "talend", "informatica", "alteryx", "sas", "sap hana", "sap", "sas enterprise",

    # --- Web & Frontend ---
    "react", "reactjs", "nextjs", "angular", "vue", "svelte",
    "bootstrap", "tailwind", "html", "css", "sass", "less",

    # --- Backend / Frameworks ---
    "django", "flask", "fastapi", "spring", "springboot",
    "express", "dotnet", "aspnet", "laravel", "rails", "symfony",

    # --- Operating Systems / Environments ---
    "linux", "windows", "macos", "ubuntu", "centos", "rhel", "debian", "android", "ios",

    # --- Version Control / CI-CD ---
    "git", "github", "gitlab", "bitbucket", "jenkins", "travisci", "circleci", "bamboo",

    # --- Cloud Security / Monitoring ---
    "splunk", "siem", "crowdstrike", "sentinelone", "paloalto", "checkpoint", "zscaler",

    # --- Collaboration / Productivity ---
    "jira", "confluence", "notion", "slack", "monday", "asana", "trello",

    # --- Miscellaneous ---
    "sap", "sap hana", "salesforce", "servicenow", "workday", "netsuite",
    "autocad", "solidworks", "unity", "unreal", "blender"
}

def remove_entities(texts, removable_entities={"ORG", "PERSON", "GPE"}): 
    cleaned = []
    for doc in ner_model.pipe(texts):
        tokens = []
        for token in doc:
            if token.ent_type_ in removable_entities and token.text.lower() not in TECH_WHITELIST:   # qui rimuovo entita'
                continue
            tokens.append(token.text_with_ws)   # attacco testo con withespaces per riottenere forma originale
        cleaned.append("".join(tokens).strip())
    return cleaned