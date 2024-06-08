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


class WordDifficultyScorer:
    def __init__(self):
        self.syllable_dict = json.load(open(Paths.WORD_TO_SYL_JSON, "r"))
        self.syllable_counter = json.load(open(Paths.SYL_TO_FREQ_JSON, "r"))
        self.SCORES = {
            "a": 1,
            "b": 3,
            "c": 3,
            "d": 2,
            "e": 1,
            "f": 4,
            "g": 2,
            "h": 4,
            "i": 1,
            "j": 8,
            "k": 5,
            "l": 1,
            "m": 3,
            "n": 1,
            "o": 1,
            "p": 3,
            "q": 10,
            "r": 1,
            "s": 1,
            "t": 1,
            "u": 1,
            "v": 4,
            "w": 4,
            "x": 8,
            "y": 4,
            "z": 10,
        }

    def total_syllable_score(self, word):
        """Assigns score based on number of syllables."""
        return len(self.syllable_dict.get(word.lower(), []))

    def unique_syllable_score(self, word):
        """Assigns score based on number of unique syllables."""
        return len(set(self.syllable_dict.get(word.lower(), [])))

    def syllable_rarity_score(self, word):
        """Assigns score based on syllable rarity. Lower value means rarer."""
        total_score = 0

        for syllable in self.syllable_dict.get(word.lower(), []):
            count = self.syllable_counter.get(syllable, 0)

            if count < 1000:
                total_score += 1
            elif count < 10000:
                total_score += 5
            else:
                total_score += 25

        return total_score

    def consonent_score(self, word):
        """Assigns difficulty score based on the number of consequent consonents."""
        max_score = score = 0

        for letter in word:
            if letter in "aeiou":
                score += 1
                max_score = max(score, max_score)
            else:
                score = 0

        return max_score

    def scrabble_score(self, word):
        """Assigns difficulty score based on how uncommon letters it contains."""
        return sum(self.SCORES.get(letter.lower(), 0) for letter in word)

    def __call__(self, df):
        df["word_scrabble_scores"] = df["words"].map(
            lambda x: np.sum([self.scrabble_score(y) for y in x])
        )
        df["word_consonent_score"] = df["words"].map(
            lambda x: np.sum([self.consonent_score(y) for y in x])
        )
        df["word_syllable_score"] = df["words"].map(
            lambda x: np.sum([self.total_syllable_score(y) for y in x])
        )
        df["word_unique_syllable_score"] = df["words"].map(
            lambda x: np.sum([self.unique_syllable_score(y) for y in x])
        )
        df["word_syllable_rarity_score"] = df["words"].map(
            lambda x: np.sum([self.syllable_rarity_score(y) for y in x])
        )
        return df


def engineer_word_features(df: pd.DataFrame, drop=True) -> pd.DataFrame:
    df = process_word(df)
    df["word_count"] = df["words"].map(lambda x: len(x))

    df = process_for_common_words(df, drop)
    df = process_for_rare_words(df, drop)
    # df = process_for_POS(df, drop)
    df = process_for_mistakes(df, drop)
    df = WordDifficultyScorer()(df)

    if drop:
        df.drop(columns=["words", "full_text", "score", "word_count"], inplace=True)

    return df
