import pandas as pd
import numpy as np

from lib.model.utils import get_score


def raw_to_class(pred, threshold):
    return pd.cut(
        pred,
        [-np.inf] + threshold + [np.inf],
        labels=[1, 2, 3, 4, 5, 6],
    ).astype(pd.Int32Dtype())


def find_thresholds(true, pred, steps=50):
    """Source:
    https://www.kaggle.com/code/cdeotte/rapids-svr-starter-cv-0-830-lb-0-804?scriptVersionId=177481746&cellId=22
    """
    # SAVE TRIALS FOR PLOTTING
    xs = [[], [], [], [], []]
    ys = [[], [], [], [], []]

    # COMPUTE BASELINE METRIC
    threshold = [1.5, 2.5, 3.5, 4.5, 5.5]
    best = get_score(true, raw_to_class(pred, threshold))

    # FIND FIVE OPTIMAL THRESHOLDS
    for k in range(5):
        for sign in [1, -1]:
            v = threshold[k]
            threshold2 = threshold.copy()
            stop = 0

            while stop < steps:
                # TRY NEW THRESHOLD
                v += sign * 0.001
                threshold2[k] = v
                metric = get_score(true, raw_to_class(pred, threshold2))

                # SAVE TRIALS FOR PLOTTING
                xs[k].append(v)
                ys[k].append(metric)

                # EARLY STOPPING
                if metric <= best:
                    stop += 1
                else:
                    stop = 0
                    best = metric
                    threshold = threshold2.copy()

    # COMPUTE FINAL METRIC
    best = get_score(true, raw_to_class(pred, threshold))

    # RETURN RESULTS
    threshold = [np.round(t, 3) for t in threshold]
    return best, threshold, xs, ys
