import re
from nltk import sent_tokenize
import numpy as np


def data_preprocessing(x: str) -> str:
    x = re.sub(r"<[^>]*>", "", x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub(r"http\S+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def process_sentence(df: pd.DataFrame) -> pd.DataFrame:
    df["sentence"] = df["full_text"].map(lambda x: sent_tokenize(x))
    df = df.explode("sentence").reset_index(drop=True)
    df["sentence"] = df["sentence"].map(data_preprocessing)
    return df


def post_process(text):
    pattern = r"sentence\s*:\s*"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        text = text[match.end() :]

    return text.strip()


def correct_sentence(model, tokenizer, device, sentences):
    sentences = [
        f"Fix grammatical errors, if any, in this sentence: {sentence}"
        for sentence in sentences
    ]
    input_ids = tokenizer(
        sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128)
    del input_ids
    corrected_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    del outputs
    return corrected_sentences


def correct_all_sentences(model, tokenizer, device, sentence_df, batch_size):
    corrected_sentences = None

    for i in range(0, sentence_df.shape[0], batch_size):
        start, end = i, i + batch_size
        sentences = sentence_df["sentence"].iloc[start:end]
        corrected = correct_sentence(model, tokenizer, device, sentences)
        corrected = np.array(corrected).flatten()

        if corrected_sentences is None:
            corrected_sentences = corrected
        else:
            corrected_sentences = np.hstack([corrected_sentences, corrected]).flatten()

    return corrected_sentences