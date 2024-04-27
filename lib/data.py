import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import pandas as pd

from .config import Config


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
        max_length=Config.MAX_LENGTH,
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
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.texts = df["full_text"].values
        self.labels = df["score"].values
        self.tokenizer = tokenizer
        self.essay_ids = df["essay_id"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "inputs": prepare_input(self.cfg, self.texts[item], self.tokenizer),
            "labels": torch.tensor(self.labels[item], dtype=torch.long),
            "essay_ids": self.essay_ids[item],
        }


def get_data_loaders(
    train_folds: pd.DataFrame,
    valid_folds: pd.DataFrame,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:

    # ======== DATASETS ==========
    train_dataset = CustomDataset(Config, train_folds, tokenizer)
    valid_dataset = CustomDataset(Config, valid_folds, tokenizer)

    # ======== DATALOADERS ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,  # TODO: split into train and valid
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE_VALID,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return (train_loader, valid_loader)
