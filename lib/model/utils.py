import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

from .deberta import DeBERTA_V3
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
    model = DeBERTA_V3(config, config_path=None, pretrained=True)
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


def load_model_from_disk(model_path, device):
    model = DeBERTA_V3(
        config,
        config_path=Paths.CONFIG_PATH,
        pretrained=False,
    )

    state = torch.load(model_path)
    model.load_state_dict(state["model"])
    model.to(device)

    return model


def quadratic_weighted_kappa(y_true, y_pred):
    """For LGBM Only"""
    y_true = y_true + config.lgbm_a
    y_pred = (y_pred + config.lgbm_a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    """For LGBM Only"""
    labels = y_true + config.lgbm_a
    preds = y_pred + config.lgbm_a
    preds = preds.clip(1, 6)
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - config.lgbm_a) ** 2 + config.lgbm_b)
    df = preds - labels
    dg = preds - config.lgbm_a
    grad = (df / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess
