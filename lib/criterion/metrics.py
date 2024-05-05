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

    precision_data, recall_data, f1_data = calcluate_metrics(df)
    plot_metrics(prefix, precision_data, recall_data, f1_data)


def calcluate_metrics(df):
    precision_data = [None for _ in range(config.num_classes)]
    recall_data = [None for _ in range(config.num_classes)]
    f1_data = [None for _ in range(config.num_classes)]

    for score in range(config.num_classes):
        tp = df.loc[(df.score == score) & (df.score == df.pred_score), "score"].count()
        fp = df.loc[(df.score != score) & (df.pred_score == score), "score"].count()
        tn = df.loc[(df.score != score) & (df.pred_score != score), "score"].count()
        fn = df.loc[(df.score == score) & (df.pred_score != score), "score"].count()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        precision_data[score] = (score, precision)
        recall_data[score] = (score, recall)
        f1_data[score] = (score, f1)

    return precision_data, recall_data, f1_data


def plot_metrics(prefix, precision_data, recall_data, f1_data):
    plot_wandb_bar(
        f1_data,
        "score",
        "F1 Score",
        f"{prefix}/F1 Score",
        f"F1 score for {prefix}",
    )
    plot_wandb_bar(
        recall_data,
        "score",
        "Recall",
        f"{prefix}/Recall",
        f"Recall for {prefix}",
    )
    plot_wandb_bar(
        precision_data,
        "score",
        "Precision",
        f"{prefix}/Precision",
        f"Precision for {prefix}",
    )


def plot_wandb_bar(data, label, value, id, title):
    table = wandb.Table(data=data, columns=[label, value])
    wandb.log({id: wandb.plot.bar(table, label, value, title=title)})
