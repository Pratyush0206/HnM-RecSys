import numpy as np

from hmrec import config
from hmrec.metrics import map_at_k


def per_customer_scores(labels, lists):
    customers = sorted(labels)
    scores = [
        map_at_k({customer: labels[customer]}, {customer: lists.get(customer, [])})
        for customer in customers
    ]
    return np.asarray(scores, dtype="float64")


def paired_bootstrap(scores_a, scores_b, n_boot=1000, seed=config.SEED, level=0.95):
    scores_a = np.asarray(scores_a, dtype="float64")
    scores_b = np.asarray(scores_b, dtype="float64")
    if scores_a.shape != scores_b.shape:
        raise ValueError("score arrays must have the same length")
    if len(scores_a) == 0:
        raise ValueError("no customers to resample")

    difference = scores_a - scores_b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(difference), size=len(difference))
        means[i] = difference[pick].mean()

    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_positive": float((means > 0).mean()),
    }