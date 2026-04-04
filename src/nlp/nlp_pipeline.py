import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nlp.nlp_extracter import extract_entities

def run_nlp_pipeline(text):
    result = extract_entities(text)
    return result