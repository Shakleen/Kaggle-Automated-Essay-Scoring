import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams

broad_operations = ["mean", "max", "sum", "min", "median"]


def jaccard_similarity(str1, str2, n=1):
    tokens1 = set(ngrams(str1.split(), n))
    tokens2 = set(ngrams(str2.split(), n))
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def cosine_distance(sentence, corrected):
    try:
        corpus = [sentence, corrected]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except:
        return 0


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = b = c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def engineer_grammar_feature(df: pd.DataFrame) -> pd.DataFrame:
    feature_list = [
        "grammar_levenshtein_distance",
        "grammar_jaccard_distance",
        "grammar_hamming_distance",
        "grammar_cosine_distance",
    ]

    df.loc[df.corrected.isna(), "corrected"] = df.loc[df.corrected.isna(), "sentence"]

    df["grammar_levenshtein_distance"] = np.vectorize(levenshtein_distance)(
        df["sentence"], df["corrected"]
    )
    df["grammar_jaccard_distance"] = np.vectorize(jaccard_similarity)(
        df["sentence"], df["corrected"]
    )
    df["grammar_hamming_distance"] = np.vectorize(hamming_distance)(
        df["sentence"], df["corrected"]
    )
    df["grammar_cosine_distance"] = np.vectorize(cosine_distance)(
        df["sentence"], df["corrected"]
    )

    feature_df = df.groupby("essay_id")[feature_list].agg(broad_operations)

    feature_df = pd.concat(
        [
            feature_df,
            df.groupby("essay_id")[feature_list]
            .agg([lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75)])
            .rename(columns={"<lambda_0>": "q1", "<lambda_1>": "q3"}),
        ],
        axis=1,
    )

    feature_df = feature_df.set_axis(feature_df.columns.map("_".join), axis=1)
    feature_df.reset_index(inplace=True)

    return feature_df
