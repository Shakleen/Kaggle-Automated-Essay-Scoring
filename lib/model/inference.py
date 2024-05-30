import torch
from torch import nn
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..data_tools.data import collate, CustomDataset
from ..config import config
from .utils import load_model_from_disk


def inference(model_path, no, weight, test_loader, device):
    softmax = nn.Softmax(dim=1)
    model = load_model_from_disk(model_path, device)
    model.eval()

    preds = None
    idx = None

    with tqdm(
        test_loader, unit="test_batch", desc=f"Model {no} Inference"
    ) as tqdm_test_loader:
        for _, batch in enumerate(tqdm_test_loader):
            inputs = collate(batch.pop("inputs"))
            ids = np.array(batch.pop("essay_ids"))

            for k, v in inputs.items():
                inputs[k] = v.to(device)  # send inputs to device

            with torch.no_grad():
                y_preds = model(inputs)  # forward propagation pass
                y_preds = softmax(y_preds.clone().detach()) * weight
                y_preds = y_preds.to("cpu").numpy().reshape(-1, config.num_classes)

            if preds is None:
                preds = y_preds
                idx = ids
            else:
                preds = np.vstack([preds, y_preds])
                idx = np.hstack([idx, ids])

    del model, softmax
    torch.cuda.empty_cache()
    gc.collect()

    preds = preds.reshape(-1, config.num_classes)
    idx = np.array(idx).flatten()

    return preds, idx


def overall_essay_score(predictions, logits=False, reduction=True):
    """Calculates the overall score from `predictions` which
    contain logits and essay ids."""
    temp = {"essay_id": predictions["essay_id"]}

    if reduction:
        for i in range(config.num_classes):
            temp[f"deberta_c{i}"] = predictions["predictions"][:, i]
    else:
        for i in range(config.n_folds):
            for j in range(config.num_classes):
                temp[f"deberta_m{i}_c{j}"] = predictions["predictions"][
                    :, j + (i * config.num_classes)
                ]

    temp = pd.DataFrame(temp)
    temp = temp.groupby("essay_id").mean().reset_index()

    if not reduction:
        return temp

    temp["score"] = np.argmax(temp.loc[:, "deberta_c0":], axis=1)
    return temp if logits else temp[["essay_id", "score"]]


def ensemble_inference(
    test_df,
    tokenizer,
    model_paths,
    device,
    logits=False,
    overall=True,
    model_wise_reduction=True,
):
    # ======== DATASETS ==========
    test_dataset = CustomDataset(config, test_df, tokenizer, is_train=False)

    # ======== DATALOADERS ==========
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_valid,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    all_preds = [None for _ in range(len(model_paths.keys()))]
    idx = None

    for i, (model_path, weight) in enumerate(model_paths.items()):
        all_preds[i], idx = inference(
            model_path,
            i,
            weight if not model_wise_reduction else 1,
            test_loader,
            device,
        )

    all_preds = np.array(all_preds)

    if model_wise_reduction:
        all_preds = np.sum(all_preds, axis=0)
    else:
        all_preds = np.hstack(all_preds)

    if overall:
        return overall_essay_score(
            {"predictions": all_preds, "essay_id": idx},
            logits,
            model_wise_reduction,
        )

    return np.argmax(all_preds, axis=1)
