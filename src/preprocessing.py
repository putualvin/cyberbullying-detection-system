import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

slangwords = {
    # Bullying & Aggressivettttt
    "kys": "kill yourself",
    "kms": "kill myself",
    "stfu": "shut the fuck up",
    "gtfo": "get the fuck out",
    "fck": "fuck",
    "fcking": "fucking",
    "fuk": "fuck",
    "bitchy": "bitch",
    "btch": "bitch",
    "noob": "newbie",
    "n00b": "newbie",
    "lame": "boring",
    "stpd": "stupid",
    "faggot": "gay",
    "trash": "bad",
    "garbage": "bad",
    "dumbass": "stupid",
    "h8": "hate",
    "luzr": "loser",

    # Pronouns & Verbs
    "u": "you",
    "ur": "your",
    "urs": "yours",
    "r": "are",
    "y": "why",
    "b": "be",
    "c": "see",
    "n": "and",
    "im": "i am",
    "ive": "i have",
    "idk": "i do not know",
    "idc": "i do not care",
    "dont": "do not",
    "cant": "can not",
    "wont": "will not",
    "aint": "is not",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",

    # Common Abbreviations (Metadata Twitter - Saran: Hapus di tahap Stopwords)
    "rt": "",
    "dm": "direct message",
    "btw": "by the way",
    "asap": "as soon as possible",
    "tbt": "throwback thursday",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "fyi": "for your information",
    "omg": "oh my god",
    "thx": "thanks",
    "tks": "thanks",
    "plz": "please",
    "pls": "please",
    "ppl": "people",
    "bc": "because",
    "b/c": "because",
    "rly": "really",
    "w/": "with",
    "w/o": "without",
    "amp": "",
    "mkr": ""
}

stop_words = set(stopwords.words('english'))

def process_input(text, use_stopwords=True):
    # Masukkan logika fix_slangwords dan clean_text kamu di sini
    # Return teks yang sudah bersih
    pass