import pandas as pd
from nltk import sent_tokenize, word_tokenize

from .utils import data_preprocessing, calculate_stats


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["paragraphs"] = df["full_text"].map(lambda x: x.split("\n\n"))
    df["paragraphs"] = df["paragraphs"].map(
        lambda x: [data_preprocessing(y) for y in x]
    )
    df.drop(columns=["full_text"], inplace=True)
    return df


def calculate_sentence_counts(df):
    df["paragraph_sentenceCount"] = df["paragraphs"].map(
        lambda x: [len(sent_tokenize(y)) for y in x]
    )
    df = calculate_stats(df, "paragraph_sentenceCount")
    df.drop(columns=["paragraph_sentenceCount"], inplace=True)
    return df


def calculate_word_counts(df):
    df["paragraph_wordCount"] = df["paragraphs"].map(
        lambda x: [len(word_tokenize(y)) for y in x]
    )
    df = calculate_stats(df, "paragraph_wordCount")
    df.drop(columns=["paragraph_wordCount"], inplace=True)
    return df


def calculate_paragraph_lengths(df):
    df["paragraph_lengths"] = df["paragraphs"].map(lambda x: [len(y) for y in x])
    df = calculate_stats(df, "paragraph_lengths")
    df.drop(columns=["paragraph_lengths"], inplace=True)
    return df


def calculate_three_part_lengths(df):
    df["paragraph_introductionLength"] = df["paragraphs"].map(lambda x: len(x[0]))
    df["paragraph_conclusionLength"] = df["paragraphs"].map(lambda x: len(x[-1]))
    df["paragraph_bodyLength"] = df["paragraphs"].map(
        lambda x: sum([len(x[i]) for i in range(1, len(x) - 1)])
    )
    return df


def engineer_paragraph_features(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(df)

    df["paragraph_count"] = df["paragraphs"].map(len)

    df = calculate_sentence_counts(df)
    df = calculate_word_counts(df)
    df = calculate_paragraph_lengths(df)
    df = calculate_three_part_lengths(df)

    return df
