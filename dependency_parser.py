import spacy
import json
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

with open("dependency_sample.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        sentence = data.get("text", "")

        doc = nlp(sentence)

        # Render dependency visualization in the browser
        print(f"\nRendering sentence: {sentence}")
        displacy.serve(doc, style="dep")
