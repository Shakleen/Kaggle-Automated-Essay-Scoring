import torch
import time
from tqdm import tqdm
import numpy as np

from .average_meter import AverageMeter
from ..config import Config
from ..data import collate
from ..utils import timeSince


def train_epoch(
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
):
    """One epoch training pass.

    Source:
    https://www.kaggle.com/code/alejopaullier/aes-2-multi-class-classification-train?scriptVersionId=170290107&cellId=26
    """
    model.train()  # set model in train mode
    scaler = torch.cuda.amp.GradScaler(
        enabled=Config.APEX
    )  # Automatic Mixed Precision tries to match each op to its appropriate datatype.
    losses = AverageMeter()  # initiate AverageMeter to track the loss.
    start = time.time()  # track the execution time.
    global_step = 0

    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc="Train") as tqdm_train_loader:
        for step, batch in enumerate(tqdm_train_loader):
            inputs = collate(batch.pop("inputs"))
            labels = batch.pop("labels")

            for k, v in inputs.items():  # send each tensor value to `device`
                inputs[k] = v.to(device)

            labels = labels.to(device)  # send labels to `device`

            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=Config.APEX):
                y_preds = model(inputs)  # forward propagation pass
                loss = criterion(y_preds, labels)  # get loss

            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

            losses.update(loss.item(), batch_size)  # update loss function tracking
            scaler.scale(loss).backward()  # backward propagation pass
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                Config.MAX_GRAD_NORM,
            )

            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)  # update optimizer parameters
                scaler.update()
                optimizer.zero_grad()  # zero out the gradients
                global_step += 1

                if Config.BATCH_SCHEDULER:
                    scheduler.step()  # update learning rate

            # ========== LOG INFO ==========
            if step % Config.PRINT_FREQ == 0 or step == (len(train_loader) - 1):
                print(
                    "Epoch: [{0}][{1}/{2}] "
                    "Elapsed {remain:s} "
                    "Loss: {loss.avg:.4f} "
                    "Grad: {grad_norm:.4f}  "
                    "LR: {lr:.8f}  ".format(
                        epoch + 1,
                        step,
                        len(train_loader),
                        remain=timeSince(start, float(step + 1) / len(train_loader)),
                        loss=losses,
                        grad_norm=grad_norm,
                        lr=scheduler.get_lr()[0],
                    )
                )

    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval()  # set model in evaluation mode
    losses = AverageMeter()  # initiate AverageMeter for tracking the loss.
    prediction_dict = {}
    preds = []
    start = time.time()  # track the execution time.

    with tqdm(valid_loader, unit="valid_batch", desc="Validation") as tqdm_valid_loader:
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

            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

            losses.update(loss.item(), batch_size)  # update loss function tracking
            preds.append(y_preds.to("cpu").numpy())  # save predictions

            # ========== LOG INFO ==========
            if step % Config.PRINT_FREQ == 0 or step == (len(valid_loader) - 1):
                print(
                    "EVAL: [{0}/{1}] "
                    "Elapsed {remain:s} "
                    "Loss: {loss.avg:.4f} ".format(
                        step,
                        len(valid_loader),
                        loss=losses,
                        remain=timeSince(start, float(step + 1) / len(valid_loader)),
                    )
                )

    prediction_dict["predictions"] = np.concatenate(
        preds
    )  # np.array() of shape (fold_size, target_cols)
    prediction_dict["essay_ids"] = student_ids
    return losses.avg, prediction_dict
