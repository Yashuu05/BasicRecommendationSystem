import spacy
import en_core_web_trf
nlp = spacy.load("en_core_web_trf")
# load the NLP neural network
nlp = en_core_web_trf.load()

def process_text(text):
    """
    function which converts a raw text to tokens
    input: raw text
    output: tokens
    """
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]
    return tokens