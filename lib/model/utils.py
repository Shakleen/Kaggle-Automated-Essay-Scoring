import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

from .deberta import CustomModel
from ..config import config
from ..paths import Paths


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return score


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]

    return optimizer_parameters


def get_scheduler(optimizer, num_train_steps):
    if config.scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=num_train_steps,
        )

    if config.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=config.num_cycles,
        )


def get_model_optimizer_and_scheduler(train_loader, device):
    model = CustomModel(config, config_path=None, pretrained=True)
    torch.save(model.config, Paths.MODEL_OUTPUT_PATH + "/config.pth")
    model.to(device)

    optimizer = AdamW(
        get_optimizer_params(
            model,
            encoder_lr=config.encoder_lr,
            decoder_lr=config.decoder_lr,
            weight_decay=config.weight_decay,
        ),
        lr=config.encoder_lr,
        eps=config.eps,
        betas=config.betas,
    )

    num_train_steps = int(len(train_loader) / config.batch_size_train * config.epochs)
    scheduler = get_scheduler(optimizer, num_train_steps)
    return model, optimizer, scheduler
