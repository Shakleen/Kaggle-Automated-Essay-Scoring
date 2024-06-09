import numpy as np
import pandas as pd
import ctypes

broad_operations = ["max", "sum", "min", "median"]


def levenshtein_distance(str1, str2):
    levenshtein = ctypes.CDLL("./lib/c_lib_compiled/levenshtein_distance.so")
    return levenshtein.levenshtein_distance(str1.encode(), str2.encode())


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
