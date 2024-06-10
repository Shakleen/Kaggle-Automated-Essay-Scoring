import pandas as pd
from nltk import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from .utils import data_preprocessing, calculate_stats


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["paragraphs"] = df["full_text"].map(lambda x: x.split("\n\n"))
    df["paragraphs"] = df["paragraphs"].map(
        lambda x: [data_preprocessing(y) for y in x]
    )
    df["sentences"] = df["paragraphs"].map(lambda x: [sent_tokenize(y) for y in x])
    df.drop(columns=["full_text"], inplace=True)
    return df


def calculate_sentence_counts(df):
    df["paragraph_sentenceCount"] = df["sentences"].map(lambda x: [len(y) for y in x])
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


def get_triple_similarity_scores(model, embeddings):
    if len(embeddings) < 3:
        return 1

    similarity_list = []

    for i in range(1, len(embeddings) - 1):
        s1 = model.similarity(embeddings[i], embeddings[i - 1])
        s2 = model.similarity(embeddings[i], embeddings[i + 1])
        mean_sim = (s1.item() + s2.item()) / 2
        similarity_list.append(mean_sim)

    return np.mean(similarity_list)


def get_dual_similarity_scores(model, embeddings):
    if len(embeddings) < 2:
        return 1

    similarity_list = []

    for i in range(1, len(embeddings)):
        s1 = model.similarity(embeddings[i], embeddings[i - 1])
        similarity_list.append(s1.item())

    return np.mean(similarity_list)


def get_pairwise_similarity_scores(model, embeddings):
    similarity_matrix = model.similarity(embeddings, embeddings)
    return torch.mean(similarity_matrix).item()


def calculate_similarity_per_paragraph(df, sentencetransformer_path: str):
    model = SentenceTransformer.load(sentencetransformer_path)
    embedding_list = [
        [model.encode(sentences) for sentences in row["sentences"]]
        for _, row in df.iterrows()
    ]
    df["paragraph_pairwiseSimilarity"] = [
        [get_pairwise_similarity_scores(model, embedding) for embedding in embeddings]
        for embeddings in embedding_list
    ]
    df["paragraph_dualSimilarity"] = [
        [get_dual_similarity_scores(model, embedding) for embedding in embeddings]
        for embeddings in embedding_list
    ]
    df["paragraph_tripleSimilarity"] = [
        [get_triple_similarity_scores(model, embedding) for embedding in embeddings]
        for embeddings in embedding_list
    ]

    del model, embedding_list

    columns = [
        "paragraph_pairwiseSimilarity",
        "paragraph_dualSimilarity",
        "paragraph_tripleSimilarity",
    ]

    for col in columns:
        df = calculate_stats(df, col)

    df.drop(columns=columns, inplace=True)
    return df


def engineer_paragraph_features(
    df: pd.DataFrame, sentencetransformer_path: str
) -> pd.DataFrame:
    df = preprocess(df)

    df["paragraph_count"] = df["paragraphs"].map(len)

    df = calculate_sentence_counts(df)
    df = calculate_word_counts(df)
    df = calculate_paragraph_lengths(df)
    df = calculate_three_part_lengths(df)
    df = calculate_similarity_per_paragraph(df, sentencetransformer_path)

    df.drop(columns=["paragraphs", "sentences"], inplace=True)

    return df
