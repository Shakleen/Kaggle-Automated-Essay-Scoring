import numpy as np
import json
import re
from nltk import sent_tokenize, word_tokenize

from .feature_engineering import data_preprocessing
from .word_feature_engineering import (
    noun_words,
    verb_words,
    pronoun_words,
    adj_words,
    adv_words,
    deter_words,
    conj_words,
    numerical_words,
    all_words,
)
from lib.paths import Paths

complex_words = set(
    [
        "because",
        "while",
        "though",
        "since",
        "if",
        "so that",
        "which",
        "who",
        "although",
        "unless",
        "when",
        "after",
        "until",
    ]
)
compound_words = set(["and", "or", "but"])
contraction_dict = json.load(open(Paths.CONTRACTION_FILE_PATH, "r"))
contraction_re = re.compile("(%s)" % "|".join(contraction_dict.keys()))


def calculate_stats(df, column):
    df[f"{column}_sum"] = df[column].map(np.sum)
    df[f"{column}_min"] = df[column].map(np.min)
    df[f"{column}_mean"] = df[column].map(np.mean)
    df[f"{column}_max"] = df[column].map(np.max)
    return df


def split_sentences(df):
    """Splits the `full_text` column into a list of sentences."""

    df["full_text"] = df["full_text"].map(data_preprocessing)
    df["sentences"] = df["full_text"].map(sent_tokenize)
    df.drop(columns=["full_text"], inplace=True)
    return df


def split_words(df):
    """Splits the words in each sentence in the column `sentences`"""

    df["sentence_words"] = df["sentences"].map(lambda x: [word_tokenize(y) for y in x])
    return df


def prepare_df(df):
    """Precalculates columns needed for feature engineering."""

    df = split_sentences(df)
    df = split_words(df)
    df["sentence_count"] = df["sentences"].map(len)
    return df


def calculate_word_mistakes_per_sentence(df):
    """Calculates the number of word mistakes per sentence."""

    df["sentence_wordMistakesPerSentence"] = df["sentence_words"].map(
        lambda sentences: [
            sum([word.lower() not in all_words for word in sentence])
            for sentence in sentences
        ]
    )
    df = calculate_stats(df, "sentence_wordMistakesPerSentence")
    df.drop(columns=["sentence_wordMistakesPerSentence"], inplace=True)

    return df


def calculate_pos_count_per_sentence(df):
    """COunts the number of individual part of speech word per sentence."""

    def count_type(sentences, type):
        return [sum([word in type for word in sentence]) for sentence in sentences]

    type_dict = {
        "Noun": noun_words,
        "Pronoun": pronoun_words,
        "Verb": verb_words,
        "Determiner": deter_words,
        "Adjective": adj_words,
        "Adverb": adv_words,
        "Numerical": numerical_words,
        "Conjunction": conj_words,
    }

    for key, val in type_dict.items():
        df[f"sentence_{key}Count"] = df["sentence_words"].map(
            lambda x: count_type(x, val)
        )
        df = calculate_stats(df, f"sentence_{key}Count")
        df.drop(columns=[f"sentence_{key}Count"], inplace=True)

    return df


def count_unique_punctuations(sentences):
    def count(sentence):
        punctuations = set()

        for token in sentence:
            for char in token:
                if char.isalnum():
                    continue

                punctuations.add(char)

        return len(punctuations)

    return [count(sentence) for sentence in sentences]


def calculate_punctuations_per_sentence(df):
    df["sentence_uniquePunctuations"] = df["sentence_words"].map(
        count_unique_punctuations
    )
    df = calculate_stats(df, "sentence_uniquePunctuations")
    df.drop(columns=["sentence_uniquePunctuations"], inplace=True)
    return df


def calculate_sentence_length(df):
    df["sentence_lengths"] = df["sentences"].map(lambda x: [len(y) for y in x])
    df = calculate_stats(df, "sentence_lengths")
    df.drop(columns=["sentence_lengths"], inplace=True)
    return df


def is_compound(words):
    if words.intersection(compound_words):
        return True
    elif ";" in words:
        return True
    return False


def is_complex(words):
    if words.intersection(complex_words):
        return True
    elif "," in words:
        return True
    return False


def classify_sentence_structure_type(sentence_words):
    sentence_type_list = ["" for _ in range(len(sentence_words))]

    for i, words in enumerate(sentence_words):
        words = set(words)

        contains_complex = is_complex(words)
        contains_compound = is_compound(words)

        if contains_complex and contains_compound:
            sentence_type_list[i] = "ComplexCompound"
        elif contains_complex:
            sentence_type_list[i] = "Complex"
        elif contains_compound:
            sentence_type_list[i] = "Compound"
        else:
            sentence_type_list[i] = "Simple"

    return sentence_type_list


def calculate_sentence_structure_ratios(df):
    """Calculates ratio of simple, complex, compound, and complex-compound sentences."""

    df["sentence_structure_type_list"] = df["sentence_words"].map(
        classify_sentence_structure_type
    )
    df["sentence_structureSimpleRatio"] = (
        df["sentence_structure_type_list"].map(
            lambda x: sum([y == "Simple" for y in x])
        )
        / df["sentence_count"]
    )
    df["sentence_structureCompoundRatio"] = (
        df["sentence_structure_type_list"].map(
            lambda x: sum([y == "Compound" for y in x])
        )
        / df["sentence_count"]
    )
    df["sentence_structureComplexRatio"] = (
        df["sentence_structure_type_list"].map(
            lambda x: sum([y == "Complex" for y in x])
        )
        / df["sentence_count"]
    )
    df["sentence_structureComplexCompoundRatio"] = (
        df["sentence_structure_type_list"].map(
            lambda x: sum([y == "ComplexCompound" for y in x])
        )
        / df["sentence_count"]
    )
    df.drop(columns=["sentence_structure_type_list"], inplace=True)
    return df


def calculate_contractions_per_sentence(df):
    df["contractions_per_sentence"] = df["sentences"].map(
        lambda x: [len(re.findall(contraction_re, sentence)) for sentence in x]
    )
    df = calculate_stats(df, "contractions_per_sentence")
    df.drop(columns=["contractions_per_sentence"], inplace=True)
    return df


def engineer_sentence_features(df):
    df = prepare_df(df)

    df = calculate_word_mistakes_per_sentence(df)
    df = calculate_pos_count_per_sentence(df)
    df = calculate_punctuations_per_sentence(df)
    df = calculate_sentence_structure_ratios(df)
    df = calculate_contractions_per_sentence(df)

    df.drop(columns=["sentences", "sentence_words"], inplace=True)
    return df
