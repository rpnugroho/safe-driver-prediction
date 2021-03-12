import numpy as np
import wandb
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert(len(actual) == len(pred))
    all = np.asarray(
        np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1*all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


gini_normalized_scorer = make_scorer(gini_normalized,
                                     needs_proba=True)


def log_cv_plot(metric, score, cv=5):
    """Helper to log and plot metrics per fold to wandb

    Args:
        metric (str): 
            metric name.
        cv (int):
            number of folds.
    """
    title = f"{metric} per fold"
    plot_id = f"fold_{metric}"
    labels = [f"{n+1}" for n in range(cv)]
    data = [[label, val]
            for (label, val) in zip(labels, score['test_'+metric])]

    table = wandb.Table(data=data, columns=['fold', metric])
    wandb.log({plot_id: wandb.plot.bar(table, 'fold', metric,
                                       title=title)})


class UpsampleStratifiedKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X, y):
            nix = np.where(y[rx] == 0)[0]
            pix = np.where(y[rx] == 1)[0]
            pixu = np.random.choice(pix, size=nix.shape[0], replace=True)
            ix = np.append(nix, pixu)
            rxm = rx[ix]
            yield rxm, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
