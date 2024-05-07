import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import pandas as pd
import re
from tqdm import tqdm

from .config import config
from .paths import Paths


def clean_text(text):
    """
    Source:
    https://www.kaggle.com/code/mpware/aes2-what-are-the-essays-about?scriptVersionId=174449147&cellId=5
    """
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_input(cfg, text, tokenizer):
    """
    This function tokenizes the input text with the configured padding and truncation. Then,
    returns the input dictionary, which contains the following keys: "input_ids",
    "token_type_ids" and "attention_mask". Each value is a torch.tensor.

    :param cfg: configuration class with a TOKENIZER attribute.
    :param text: a numpy array where each value is a text as string.
    :return inputs: python dictionary where values are torch tensors.

    Source:
    https://www.kaggle.com/code/alejopaullier/aes-2-multi-class-classification-train?scriptVersionId=170290107&cellId=18
    """
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=config.max_length,
        padding="max_length",  # TODO: check padding to max sequence in batch
        truncation=True,
    )

    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)  # TODO: check dtypes

    return inputs


def collate(inputs):
    """
    It truncates the inputs to the maximum sequence length in the batch.

    Source:
    https://www.kaggle.com/code/alejopaullier/aes-2-multi-class-classification-train?scriptVersionId=170290107&cellId=18
    """
    mask_len = int(
        inputs["attention_mask"].sum(axis=1).max()
    )  # Get batch's max sequence length

    for k, _ in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]

    return inputs


class CustomDataset(Dataset):
    """
    Source:
    https://www.kaggle.com/code/alejopaullier/aes-2-multi-class-classification-train?scriptVersionId=170290107&cellId=18
    """

    def __init__(self, cfg, df, tokenizer, is_train: bool = True):
        self.cfg = cfg
        self.texts = df["full_text"].values
        self.essay_ids = df["essay_id"].values
        self.tokenizer = tokenizer
        self.is_train = is_train

        if self.is_train:
            self.labels = df["score"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        output = {
            "inputs": prepare_input(self.cfg, self.texts[item], self.tokenizer),
            "essay_ids": self.essay_ids[item],
        }

        if self.is_train:
            output["labels"] = torch.tensor(self.labels[item], dtype=torch.long)

        return output


def get_data_loaders(
    train_folds: pd.DataFrame,
    valid_folds: pd.DataFrame,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:

    # ======== DATASETS ==========
    train_dataset = CustomDataset(config, train_folds, tokenizer)
    valid_dataset = CustomDataset(config, valid_folds, tokenizer)

    # ======== DATALOADERS ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,  # TODO: split into train and valid
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size_valid,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return (train_loader, valid_loader)


def read_data_loader_from_disk(
    fold: int, group: int = None
) -> Tuple[DataLoader, DataLoader]:
    """Reads train and valid data loader for fold `fold` from disk.

    Args:
        fold (int): Fold number.
        group (int): Group number when

    Returns:
        Tuple[DataLoader, DataLoader]: Train and valid data loader.
    """
    if group is None:
        train_file_name = f"train_{fold}_{group}.pth"
        valid_file_name = f"valid_{fold}_{group}.pth"
    else:
        train_file_name = f"train_{fold}.pth"
        valid_file_name = f"valid_{fold}.pth"

    train_loader = torch.load(os.path.join(Paths.DATA_LOADER_PATH, train_file_name))
    valid_loader = torch.load(os.path.join(Paths.DATA_LOADER_PATH, valid_file_name))
    return (train_loader, valid_loader)


def split_tokens(tokens, stride=config.stride):
    """Splits `tokens` into multiple sequences that have at most
    `config.max_length` tokens. Uses `config.stride` for sliding
    window.

    Args:
        tokens (List): List of tokens.
        stride (int): Stride length. (Default: config.stride)

    Returns:
        List[List[int]]: List of split token sequences.
    """
    start = 0
    sequence_list = []

    while start < len(tokens):
        remaining_tokens = len(tokens) - start

        if remaining_tokens < config.max_length and start > 0:
            start = max(0, len(tokens) - config.max_length)

        end = min(start + config.max_length, len(tokens))
        sequence_list.append(tokens[start:end])

        if remaining_tokens >= config.max_length:
            start += stride
        else:
            break

    return sequence_list


def _construct_new_row(old_row, text):
    new_row = {key: old_row[key] for key in old_row.keys() if key != "index"}
    new_row["full_text"] = text
    return new_row


def sliding_window(df, tokenizer):
    """Splits rows of `df` so that each row's text has at most
    `config.max_length` number of tokens.

    Args:
        df (pd.DataFrame): Input data frame.
        tokenizer (_type_): Tokenizer used to encode and decode text.

    Returns:
        pd.DataFrame: Newly constructed dataframe.
    """
    new_df = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        tokens = tokenizer.encode(row["full_text"], add_special_tokens=False)

        if len(tokens) <= config.max_length:
            new_df.append(_construct_new_row(row, row["full_text"]))
        else:
            sequence_list = split_tokens(tokens, get_stride_value(row))

            for seq in sequence_list:
                new_df.append(
                    _construct_new_row(
                        row,
                        tokenizer.decode(seq, skip_special_tokens=True),
                    )
                )

    return pd.DataFrame(new_df)


def get_stride_value(row):
    if not config.oversample:
        return config.stride

    # Oversample scores 1 and 6
    if row["score"] == 0:
        return config.stride // 2
    elif row["score"] == 5:
        return config.stride // 3

    # The rest are unchanged
    return config.stride
