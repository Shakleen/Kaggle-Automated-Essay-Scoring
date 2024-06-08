import re
import os
import pandas as pd
import json
from nltk.tokenize import word_tokenize
import numpy as np

from lib.paths import Paths


def get_word_list_from_file(file_name):
    file_path = os.path.join(Paths.WORD_LIST_DIR, file_name)

    with open(file_path) as file:
        output = set([re.sub("\n", "", word) for word in file.readlines()])

    return output


noun_words = get_word_list_from_file("top_english_nouns_mixed_10000.txt")
verb_words = get_word_list_from_file("top_english_verbs_mixed_10000.txt")
pronoun_words = get_word_list_from_file("top_english_prons_mixed_10000.txt")
adj_words = get_word_list_from_file("top_english_adjs_mixed_10000.txt")
adv_words = get_word_list_from_file("top_english_advs_mixed_10000.txt")
deter_words = get_word_list_from_file("top_english_dets_lower_500.txt")
conj_words = get_word_list_from_file("top_english_conjs_lower_500.txt")
numerical_words = get_word_list_from_file("top_english_nums_lower_500.txt")
all_words = (
    noun_words
    | verb_words
    | pronoun_words
    | adj_words
    | adj_words
    | deter_words
    | conj_words
    | numerical_words
)
common_words = get_word_list_from_file("english_most_common_5000.txt")
rare_words = get_word_list_from_file("google-10000-english-no-swears.txt")
contraction_dict = json.load(open(Paths.CONTRACTION_FILE_PATH, "r"))
contraction_re = re.compile("(%s)" % "|".join(contraction_dict.keys()))


def expand_contractions(text: str, c_re=contraction_re) -> str:
    """Replaces contracted word/phrase with enlongated word/phrase."""

    def replace(match):
        return contraction_dict[match.group(0)]

    return c_re.sub(replace, text)


def data_preprocessing(x: str) -> str:
    x = re.sub(r"<[^>]*>", "", x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub(r"http\S+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", " ", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub("-|`", " ", x)
    x = x.strip()
    return x


def process_word(df: pd.DataFrame) -> pd.DataFrame:
    temp = df["full_text"].map(expand_contractions)
    temp = temp.map(data_preprocessing)
    df["words"] = temp.map(word_tokenize)
    return df


def process_for_common_words(df: pd.DataFrame, drop: bool) -> pd.DataFrame:
    df["common"] = df["words"].map(lambda x: [y for y in x if y in common_words])
    df["word_common_percentage"] = df["common"].map(lambda x: len(x)) / df["word_count"]

    if drop:
        df.drop(columns=["common"], inplace=True)

    return df


def process_for_rare_words(df: pd.DataFrame, drop: bool) -> pd.DataFrame:
    df["rare"] = df["words"].map(lambda x: [y for y in x if y in rare_words])
    df["word_rare_percentage"] = df["rare"].map(lambda x: len(x)) / df["word_count"]

    if drop:
        df.drop(columns=["rare"], inplace=True)

    return df


# Not worth it
# def process_for_POS(df: pd.DataFrame, drop: bool) -> pd.DataFrame:
#     """
#     List from this repo:
#     https://github.com/david47k/top-english-wordlists?tab=readme-ov-file
#     """
#     df["noun"] = df["words"].map(lambda x: [y for y in x if y in noun_words])
#     df["verb"] = df["words"].map(lambda x: [y for y in x if y in verb_words])
#     df["pronoun"] = df["words"].map(lambda x: [y for y in x if y in pronoun_words])
#     df["adjective"] = df["words"].map(lambda x: [y for y in x if y in adj_words])
#     df["adverb"] = df["words"].map(lambda x: [y for y in x if y in adv_words])
#     df["determiner"] = df["words"].map(lambda x: [y for y in x if y in deter_words])
#     df["conjunction"] = df["words"].map(lambda x: [y for y in x if y in conj_words])
#     df["numerical"] = df["words"].map(lambda x: [y for y in x if y in numerical_words])

#     df["word_noun_percentage"] = df["noun"].map(lambda x: len(x)) / df["word_count"]
#     df["word_verb_percentage"] = df["verb"].map(lambda x: len(x)) / df["word_count"]
#     df["word_pronoun_percentage"] = (
#         df["pronoun"].map(lambda x: len(x)) / df["word_count"]
#     )
#     df["word_adjective_percentage"] = (
#         df["adjective"].map(lambda x: len(x)) / df["word_count"]
#     )
#     df["word_adverb_percentage"] = df["adverb"].map(lambda x: len(x)) / df["word_count"]
#     df["word_determiner_percentage"] = (
#         df["determiner"].map(lambda x: len(x)) / df["word_count"]
#     )
#     df["word_conjunction_percentage"] = (
#         df["conjunction"].map(lambda x: len(x)) / df["word_count"]
#     )
#     df["word_numerical_percentage"] = (
#         df["numerical"].map(lambda x: len(x)) / df["word_count"]
#     )

#     if drop:
#         df.drop(
#             columns=[
#                 "noun",
#                 "verb",
#                 "pronoun",
#                 "adjective",
#                 "adverb",
#                 "determiner",
#                 "conjunction",
#                 "numerical",
#             ],
#             inplace=True,
#         )

#     return df


def process_for_mistakes(df: pd.DataFrame, drop: bool) -> pd.DataFrame:
    df["mistakes"] = df["words"].map(lambda x: set(x).difference(all_words))

    df["word_mistake_percentage"] = (
        df["mistakes"].map(lambda x: len(x)) / df["word_count"]
    )

    if drop:
        df.drop(columns=["mistakes"], inplace=True)

    return df


# TODO: Find a way to implement without library usage
# def process_for_difficult_words(df: pd.DataFrame, drop: bool) -> pd.DataFrame:
#     """
#     Credit to this repo by andrei-papou
#     https://github.com/andrei-papou/textstat
#     """
#     df["word_difficult_percentage"] = (
#         df["full_text"].map(textstat.difficult_words) / df["word_count"]
#     )

#     if drop:
#         df.drop(columns=["full_text"], inplace=True)

#     return df


def calculate_percentages(df, drop):
    df = process_word(df)
    df["word_count"] = df["words"].map(lambda x: len(x))

    # df = process_for_difficult_words(df, drop)
    df = process_for_common_words(df, drop)
    df = process_for_rare_words(df, drop)
    # df = process_for_POS(df, drop)
    df = process_for_mistakes(df, drop)

    if drop:
        df.drop(columns=["words"], inplace=True)

    return df


def engineer_word_features(df: pd.DataFrame, drop=True) -> pd.DataFrame:
    df = calculate_percentages(df, drop)

    broad_operations = ["mean", "max", "sum"]
    feature_list = [
        "word_common_percentage",
        "word_rare_percentage",
        # "word_noun_percentage",
        # "word_verb_percentage",
        # "word_pronoun_percentage",
        # "word_adjective_percentage",
        # "word_adverb_percentage",
        # "word_determiner_percentage",
        # "word_conjunction_percentage",
        # "word_numerical_percentage",
        "word_mistake_percentage",
        # "word_difficult_percentage",
    ]

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
