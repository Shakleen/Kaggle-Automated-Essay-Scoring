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


def calculate_stats(df, column):
    df[f"{column}_sum"] = df[column].map(np.sum)
    df[f"{column}_min"] = df[column].map(np.min)
    df[f"{column}_mean"] = df[column].map(np.mean)
    df[f"{column}_max"] = df[column].map(np.max)
    return df


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
    temp = df["full_text"].map(data_preprocessing)
    df["words"] = temp.map(word_tokenize)
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
        df["word_scrabbleScores"] = df["words"].map(
            lambda x: [self.scrabble_score(y) for y in x]
        )
        df["word_consonentScores"] = df["words"].map(
            lambda x: [self.consonent_score(y) for y in x]
        )
        df["word_syllableScores"] = df["words"].map(
            lambda x: [self.total_syllable_score(y) for y in x]
        )
        df["word_uniqueSyllableScores"] = df["words"].map(
            lambda x: [self.unique_syllable_score(y) for y in x]
        )
        df["word_syllableRarityScores"] = df["words"].map(
            lambda x: [self.syllable_rarity_score(y) for y in x]
        )

        columns = [
            "word_scrabbleScores",
            "word_consonentScores",
            "word_syllableScores",
            "word_uniqueSyllableScores",
            "word_syllableRarityScores",
        ]

        for column in columns:
            df = calculate_stats(df, column)

        df.drop(columns=columns, inplace=True)

        return df


def engineer_word_features(df: pd.DataFrame) -> pd.DataFrame:
    df = process_word(df)
    df["word_count"] = df["words"].map(lambda x: len(x))
    df["word_variety"] = df["words"].map(
        lambda x: len(set(y for y in x if y in all_words))
    )

    df = WordDifficultyScorer()(df)

    df.drop(columns=["words", "full_text"], inplace=True)

    return df
