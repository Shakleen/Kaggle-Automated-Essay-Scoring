import torch
from torch import nn


class RMSELoss(nn.Module):
    """Source: https://www.kaggle.com/code/alejopaullier/aes-2-multi-class-classification-train?scriptVersionId=170290107&cellId=24"""

    def __init__(self, reduction="mean", eps=1e-9):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)

        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss
