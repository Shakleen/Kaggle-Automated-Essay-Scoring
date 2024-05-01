import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import pandas as pd

from .config import config
from .paths import Paths


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


def read_data_loader_from_disk(fold: int) -> Tuple[DataLoader, DataLoader]:
    """Reads train and valid data loader for fold `fold` from disk.

    Args:
        fold (int): Fold number.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and valid data loader.
    """
    train_loader = torch.load(os.path.join(Paths.DATA_LOADER_PATH, f"train_{fold}.pth"))
    valid_loader = torch.load(os.path.join(Paths.DATA_LOADER_PATH, f"valid_{fold}.pth"))
    return (train_loader, valid_loader)
