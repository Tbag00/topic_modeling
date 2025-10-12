import spacy
from spacy.tokens import DocBin
import json
from pathlib import Path

nlp = spacy.blank("en")
db = DocBin()
wd = Path(__file__).parent

with open(wd.parent / "jsonl" / "doccano_export_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        text = record["text"]
        doc = nlp.make_doc(text)

        spans = []
        for start, end, label in record.get("label", []):
            span = doc.char_span(int(start), int(end), label=label)
            if span is not None:
                spans.append(span)
        doc.spans["sc"] = spans  # il tuo spans_key nel config
        db.add(doc)

db.to_disk(wd / "test.spacy")