import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nlp.constants import ISSUE_KEYWORDS, DEVICE_KEYWORDS, SEVERITY_KEYWORDS

def match_keyword(text, keyword_dict):
    """
    This function returns macthed kwyword 
    input: raw text (e.g: broken screen of laptop)
    output: matched keyword (broken, laptop)
    """
    for label, keyword in keyword_dict.items():
        for key in keyword:
            if key in text.lower():
                return label
    return None

def detect_urgency(text):
    urgent_words = ["urgent", "asap", "immediately", "fast","immediate"]

    for word in urgent_words:
        if word in text:
            return "yes"
    return "no"

def extract_entities(text):

    """
    this function returns dictionary of extracted keywords
    input: raw text
    output: dictionary of extracted keywords
    """
    text = text.lower()
    issue = match_keyword(text, ISSUE_KEYWORDS)
    device = match_keyword(text, DEVICE_KEYWORDS)
    severity = match_keyword(text, SEVERITY_KEYWORDS)
    urgency = detect_urgency(text)

    return {
        "issue":issue,
        "device": device,
        "severity": severity,
        "urgency": urgency
    }
