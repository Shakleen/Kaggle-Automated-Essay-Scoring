import spacy
import re
import numpy as np

from lib.paths import Paths


def calculate_stats(df, column):
    df[f"{column}_sum"] = df[column].map(np.sum)
    df[f"{column}_min"] = df[column].map(np.min)
    df[f"{column}_mean"] = df[column].map(np.mean)
    df[f"{column}_max"] = df[column].map(np.max)
    return df


nlp = spacy.load("en_core_web_sm")

with open(Paths.ENG_WORDS_HX, "r") as file:
    english_vocab = set(word.strip().lower() for word in file)


def count_spelling_errors(text):
    """Uses `spacy` and list of correctly spelled english words
    located at `Paths.ENG_WORDS_HX` to count number of spelling
    errors.
    """
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_.lower() for token in doc]

    spelling_errors = sum(
        1 for token in lemmatized_tokens if token not in english_vocab
    )

    return spelling_errors


def remove_HTML_tags(text: str) -> str:
    """Remove HTML tags from a text string"""
    return re.sub(r"<[^>]*>", "", text)


def remove_URL(text: str) -> str:
    """Remove URLs from a text string"""
    return re.sub(r"http\S+", "", text)


def data_preprocessing(x: str) -> str:
    x = x.lower()
    x = remove_HTML_tags(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = remove_URL(x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x
