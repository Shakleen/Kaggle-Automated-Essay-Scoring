import wandb

from ..config import config


def log_metrics(df, prefix):
    # Confusion Matrix
    wandb.log(
        {
            f"{prefix}/conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=df["score"],
                preds=df["pred_score"],
                class_names=list(
                    range(config.num_classes),
                ),
            )
        }
    )

    # Other metrics
    for score in range(config.num_classes):
        tp = df.loc[(df.score == score) & (df.score == df.pred_score), "score"].count()
        fp = df.loc[(df.score != score) & (df.pred_score == score), "score"].count()
        tn = df.loc[(df.score != score) & (df.pred_score != score), "score"].count()
        fn = df.loc[(df.score == score) & (df.pred_score != score), "score"].count()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        error_rate = 1 - accuracy
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        sensitivity = recall
        specificity = tn / (tn + fp)

        wandb.log(
            {
                f"{prefix}/Accuracy": accuracy,
                f"{prefix}/Error Rate": error_rate,
                f"{prefix}/Precision": precision,
                f"{prefix}/Recall": recall,
                f"{prefix}/F1 Score": f1,
                f"{prefix}/Sensitivity": sensitivity,
                f"{prefix}/Specificity": specificity,
            }
        )
