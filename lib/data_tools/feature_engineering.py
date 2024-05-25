import spacy
import re
import json
import string
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats import kurtosis

from lib.paths import Paths

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


contraction_dict = json.load(open(Paths.CONTRACTION_FILE_PATH, "r"))
contraction_re = re.compile("(%s)" % "|".join(contraction_dict.keys()))


def expand_contractions(text: str, c_re=contraction_re) -> str:
    """Replaces contracted word/phrase with enlongated word/phrase."""

    def replace(match):
        return contraction_dict[match.group(0)]

    return c_re.sub(replace, text)


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


def remove_punctuation(text: str) -> str:
    """A translator is created using str.maketrans('', '', string.punctuation),
    which generates a translation table that maps each character in the
    string string.punctuation to None. This effectively removes all punctuation characters.
    """
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def process_paragraph(df: pd.DataFrame) -> pd.DataFrame:
    # Construct paragraphs
    df["paragraph"] = df["full_text"].map(lambda x: x.split("\n\n"))

    # Have each paragraph be its own row
    df = df.explode("paragraph")

    # Process Paragraph text
    df["paragraph"] = df["paragraph"].map(data_preprocessing)
    df["paragraph_no_punctuation"] = df["paragraph"].map(remove_punctuation)

    # Calculate base stats
    df["paragraph_error_count"] = df["paragraph_no_punctuation"].map(
        count_spelling_errors
    )
    df["paragraph_char_count"] = df["paragraph"].map(lambda x: len(x))
    df["paragraph_word_count"] = df["paragraph"].map(
        lambda x: len(re.findall(r"\w+", x))
    )
    df["paragraph_sentence_count"] = df["paragraph"].map(
        lambda x: len(re.findall(r"[.!?]", x))
    )

    return df


def calculate_length_features(df, feature_name, length_range):
    feature_df = pd.DataFrame()

    for l in length_range:
        temp = (
            df.groupby("essay_id")[feature_name]
            .agg([lambda x: len(x) < l, lambda x: len(x) >= l])
            .rename(columns={"<lambda_0>": f"len<{l}", "<lambda_1>": f"len>={l}"})
        ).reset_index(drop=True)
        feature_df = pd.concat([feature_df, temp], axis=1)

    return feature_df


def paragraph_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    feature_list = [
        "paragraph_error_count",
        "paragraph_char_count",
        "paragraph_word_count",
        "paragraph_sentence_count",
    ]

    feature_df = df.groupby("essay_id")[feature_list].agg(
        ["mean", "min", "max", "sum", "first", "last"]
    )

    feature_df = pd.concat(
        [
            feature_df,
            df.groupby("essay_id")[feature_list]
            .agg(
                [
                    lambda x: np.quantile(x, 0.25),
                    lambda x: np.quantile(x, 0.75),
                    lambda x: kurtosis(x),
                ]
            )
            .rename(
                columns={
                    "<lambda_0>": "q1",
                    "<lambda_1>": "q3",
                    "<lambda_2>": "kurtosis",
                }
            ),
        ],
        axis=1,
    )

    for name, lengths in [
        ("paragraph_char_count", range(50, 701, 50)),
        ("paragraph_word_count", range(50, 501, 50)),
        ("paragraph_sentence_count", range(5, 51, 5)),
        ("paragraph_error_count", range(5, 51, 5)),
    ]:
        feature_df = pd.concat(
            [feature_df, calculate_length_features(df, name, lengths)],
            axis=1,
        )

    feature_df = feature_df.set_axis(feature_df.columns.map("_".join), axis=1)
    feature_df.reset_index(inplace=True)
    feature_df.rename(columns={"index": "essay_id"}, inplace=True)
    return feature_df


def process_sentence(df: pd.DataFrame) -> pd.DataFrame:
    # Construct sentences
    df["sentence"] = df["full_text"].map(lambda x: sent_tokenize(x))

    # Have each paragraph be its own row
    df = df.explode("sentence")

    # Process Paragraph text
    df["sentence"] = df["sentence"].map(data_preprocessing)
    df["sentence_no_punctuation"] = df["sentence"].map(remove_punctuation)

    # Calculate base stats
    df["sentence_error_count"] = df["sentence_no_punctuation"].map(
        count_spelling_errors
    )
    df["sentence_char_count"] = df["sentence"].map(lambda x: len(x))
    df["sentence_word_count"] = df["sentence"].map(lambda x: len(re.findall(r"\w+", x)))

    return df


def sentence_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    feature_list = [
        "sentence_error_count",
        "sentence_char_count",
        "sentence_word_count",
    ]

    feature_df = df.groupby("essay_id")[feature_list].agg(
        ["mean", "min", "max", "sum", "first", "last"]
    )

    feature_df = pd.concat(
        [
            feature_df,
            df.groupby("essay_id")[feature_list]
            .agg(
                [
                    lambda x: np.quantile(x, 0.25),
                    lambda x: np.quantile(x, 0.75),
                    lambda x: kurtosis(x),
                ]
            )
            .rename(
                columns={
                    "<lambda_0>": "q1",
                    "<lambda_1>": "q3",
                    "<lambda_2>": "kurtosis",
                }
            ),
        ],
        axis=1,
    )

    for name, lengths in [
        ("sentence_char_count", range(25, 301, 25)),
        ("sentence_word_count", range(5, 51, 5)),
        ("sentence_error_count", range(2, 11, 2)),
    ]:
        feature_df = pd.concat(
            [feature_df, calculate_length_features(df, name, lengths)],
            axis=1,
        )

    feature_df = feature_df.set_axis(feature_df.columns.map("_".join), axis=1)
    feature_df.reset_index(inplace=True)
    feature_df.rename(columns={"index": "essay_id"}, inplace=True)
    return feature_df


def process_word(df: pd.DataFrame) -> pd.DataFrame:
    # Get words
    temp = df["full_text"].map(data_preprocessing)
    df["word"] = temp.map(lambda x: x.split(" "))

    # Have each paragraph be its own row
    df = df.explode("word")

    # Calculate base stats
    df["word_char_count"] = df["word"].map(lambda x: len(x))

    return df


def word_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    feature_list = ["word_char_count"]

    feature_df = df.groupby("essay_id")[feature_list].agg(["mean", "min", "max"])

    feature_df = pd.concat(
        [
            feature_df,
            df.groupby("essay_id")[feature_list]
            .agg(
                [
                    lambda x: np.quantile(x, 0.25),
                    lambda x: np.quantile(x, 0.50),
                    lambda x: np.quantile(x, 0.75),
                ]
            )
            .rename(
                columns={
                    "<lambda_0>": "q1",
                    "<lambda_1>": "q2",
                    "<lambda_1>": "q3",
                }
            ),
        ],
        axis=1,
    )

    for l in range(5, 31, 2):
        temp = (
            df.groupby("essay_id")[feature_list]
            .agg([lambda x: len(x) < l, lambda x: len(x) >= l])
            .rename(columns={"<lambda_0>": f"len<{l}", "<lambda_1>": f"len>={l}"})
        ).reset_index(drop=True)
        feature_df = pd.concat([feature_df, temp], axis=1)

    feature_df = feature_df.set_axis(feature_df.columns.map("_".join), axis=1)
    feature_df.reset_index(inplace=True)
    feature_df.rename(columns={"index": "essay_id"}, inplace=True)
    return feature_df


# Need to have this separately, otherwise can't pickle
# Source:
# https://datascience.stackexchange.com/questions/67189/unable-to-save-the-tf-idf-vectorizer
def func(x):
    return x


def generate_tfidf_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer = None,
) -> pd.DataFrame:
    if not vectorizer:
        vectorizer = TfidfVectorizer(
            tokenizer=func,
            preprocessor=func,
            token_pattern=None,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(3, 6),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
        )

        tfidf_features = vectorizer.fit_transform([i for i in df["full_text"]])
    else:
        tfidf_features = vectorizer.transform([i for i in df["full_text"]])

    tfidf_features = pd.DataFrame(tfidf_features.toarray())
    tfidf_features.columns = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
    tfidf_features["essay_id"] = df["essay_id"].copy()

    return vectorizer, tfidf_features


def generate_count_features(
    df: pd.DataFrame,
    vectorizer_cnt: CountVectorizer = None,
) -> pd.DataFrame:
    if not vectorizer_cnt:
        vectorizer_cnt = CountVectorizer(
            tokenizer=func,
            preprocessor=func,
            token_pattern=None,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(2, 3),
            min_df=0.10,
            max_df=0.85,
        )

        count_features = vectorizer_cnt.fit_transform([i for i in df["full_text"]])
    else:
        count_features = vectorizer_cnt.transform([i for i in df["full_text"]])

    count_features = pd.DataFrame(count_features.toarray())
    count_features.columns = [
        f"tfidf_count_{i}" for i in range(count_features.shape[1])
    ]
    count_features["essay_id"] = df["essay_id"].copy()

    return vectorizer_cnt, count_features
