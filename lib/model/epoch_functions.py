import torch
import time
from tqdm import tqdm
import numpy as np
import wandb
import math

from ..utils.average_meter import AverageMeter
from ..config import config
from ..data import collate
from ..utils.utils import timeSince


def train_epoch(
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    group=None,
):
    """One epoch training pass.

    Source:
    https://www.kaggle.com/code/alejopaullier/aes-2-multi-class-classification-train?scriptVersionId=170290107&cellId=26
    """
    model.train()  # set model in train mode
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.apex
    )  # Automatic Mixed Precision tries to match each op to its appropriate datatype.
    losses = AverageMeter()  # initiate AverageMeter to track the loss.
    global_step = 0
    n_steps_per_epoch = math.ceil(len(train_loader) / config.batch_size_train)

    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(
        train_loader, unit="train_batch", desc=_get_tqdm_desc(fold, group)
    ) as tqdm_train_loader:
        for step, batch in enumerate(tqdm_train_loader):
            inputs = collate(batch.pop("inputs"))
            labels = batch.pop("labels")

            for k, v in inputs.items():  # send each tensor value to `device`
                inputs[k] = v.to(device)

            labels = labels.to(device)  # send labels to `device`

            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=config.apex):
                y_preds = model(inputs)  # forward propagation pass
                loss = criterion(y_preds, labels)  # get loss

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            losses.update(loss.item(), batch_size)  # update loss function tracking
            scaler.scale(loss).backward()  # backward propagation pass
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)  # update optimizer parameters
                scaler.update()
                optimizer.zero_grad()  # zero out the gradients
                global_step += 1

                if config.batch_scheduler:
                    scheduler.step()  # update learning rate

            # ========== LOG INFO ==========
            if step % config.print_freq == 0 or step == (len(train_loader) - 1):
                _log_training_metrics(
                    fold,
                    epoch,
                    scheduler,
                    group,
                    losses,
                    n_steps_per_epoch,
                    step,
                    grad_norm,
                )

    return losses.avg


def _log_training_metrics(
    fold, epoch, scheduler, group, losses, n_steps_per_epoch, step, grad_norm
):
    if group is None:
        wandb.log(
            {
                f"train/epoch_f{fold}": calc_epoch(epoch, n_steps_per_epoch, step),
                f"train/train_loss_f{fold}": losses.avg,
                f"train/grad_norm_f{fold}": grad_norm,
                f"train/learning_rate_f{fold}": scheduler.get_lr()[0],
            }
        )
    else:
        wandb.log(
            {
                f"train/epoch_f{fold}_g{group}": calc_epoch(
                    epoch, n_steps_per_epoch, step
                ),
                f"train/train_loss_f{fold}_g{group}": losses.avg,
                f"train/grad_norm_f{fold}_g{group}": grad_norm,
                f"train/learning_rate_f{fold}_g{group}": scheduler.get_lr()[0],
            }
        )


def _get_tqdm_desc(fold, group):
    desc = f"Training Fold {fold}"

    if group:
        desc = f"{desc} Group {group}"
    return desc


def calc_epoch(epoch, n_steps_per_epoch, step):
    return (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch


def valid_epoch(fold, valid_loader, model, criterion, device):
    model.eval()  # set model in evaluation mode
    losses = AverageMeter()  # initiate AverageMeter for tracking the loss.
    prediction_dict = {}
    preds = []

    with tqdm(
        valid_loader, unit="valid_batch", desc=f"Validating Fold {fold}"
    ) as tqdm_valid_loader:
        for step, batch in enumerate(tqdm_valid_loader):
            inputs = collate(batch.pop("inputs"))  # collate inputs
            labels = batch.pop("labels")
            student_ids = batch.pop("essay_ids")

            for k, v in inputs.items():
                inputs[k] = v.to(device)  # send inputs to device

            labels = labels.to(device)

            batch_size = labels.size(0)
            with torch.no_grad():
                y_preds = model(inputs)  # forward propagation pass
                loss = criterion(y_preds, labels)  # get loss

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            losses.update(loss.item(), batch_size)  # update loss function tracking
            preds.append(y_preds.to("cpu").numpy())  # save predictions

    prediction_dict["predictions"] = np.concatenate(
        preds
    )  # np.array() of shape (fold_size, target_cols)
    prediction_dict["essay_ids"] = student_ids
    return losses.avg, prediction_dict
