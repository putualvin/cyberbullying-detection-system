import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords jika belum ada
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup Stopwords persis seperti di Notebook
stop_words = set(stopwords.words("english"))
keep_words = {"no", "nor", "not", "dont", "don't", "cant", "can't", "won't", "shouldn't", "isn't", "aren't"}
stop_words = stop_words - keep_words

# Kamus Slangwords
slangwords = {
    "kys": "kill yourself", "kms": "kill myself", "stfu": "shut the fuck up",
    "gtfo": "get the fuck out", "fck": "fuck", "fcking": "fucking", "fuk": "fuck",
    "bitchy": "bitch", "btch": "bitch", "noob": "newbie", "n00b": "newbie",
    "lame": "boring", "stpd": "stupid", "faggot": "gay", "trash": "bad",
    "garbage": "bad", "dumbass": "stupid", "h8": "hate", "luzr": "loser",
    "u": "you", "ur": "your", "urs": "yours", "r": "are", "y": "why",
    "b": "be", "c": "see", "n": "and", "im": "i am", "ive": "i have",
    "idk": "i do not know", "idc": "i do not care", "dont": "do not",
    "cant": "can not", "wont": "will not", "aint": "is not",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "rt": "", "dm": "direct message", "btw": "by the way", "asap": "as soon as possible",
    "tbt": "throwback thursday", "lol": "laughing out loud", "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing", "fyi": "for your information",
    "omg": "oh my god", "thx": "thanks", "tks": "thanks", "plz": "please",
    "pls": "please", "ppl": "people", "bc": "because", "b/c": "because",
    "rly": "really", "w/": "with", "w/o": "without", "amp": "", "mkr": ""
}

def fix_slangwords(text):
    if not isinstance(text, str):
        return ""
    words = text.split()
    fixed_words = [slangwords.get(w.lower(), w) for w in words if slangwords.get(w.lower(), w) != ""]
    return " ".join(fixed_words)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def process_input(text):
    # Fix untuk error NoneType
    if not text or not str(text).strip():
        return ""
    
    text_slang_fixed = fix_slangwords(text)
    final_text = clean_text(text_slang_fixed)
    return final_text