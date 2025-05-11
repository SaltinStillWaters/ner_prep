import spacy
from spacy.language import Language
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load your fine-tuned Hugging Face NER model
model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create the NER pipeline
hf_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Define custom spaCy component using Language.factory
@Language.factory("hf_ner_component")
def create_hf_ner_component(nlp, name):
    def hf_ner_component(doc):
        results = hf_ner(doc.text)
        ents = []
        for r in results:
            start_char = r["start"]
            end_char = r["end"]
            label = r["entity_group"]
            span = doc.char_span(start_char, end_char, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        return doc
    return hf_ner_component

# Build spaCy pipeline
nlp = spacy.blank("en")
nlp.add_pipe("hf_ner_component", last=True)

# Run inference
doc = nlp("square <expression>")
print([(ent.text, ent.label_) for ent in doc.ents])
