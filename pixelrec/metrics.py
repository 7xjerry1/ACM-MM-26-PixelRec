import numpy as np


def full_sort_metrics(answers, predictions, ks=(5, 10, 20)):
    answers = np.asarray(answers).reshape(-1, 1)
    predictions = np.asarray(predictions)
    result = {}
    for k in ks:
        topk = predictions[:, :k]
        hits = topk == answers
        result[f"H@{k}"] = float(hits.any(axis=1).mean())
        ranks = np.argmax(hits, axis=1) + 1
        gains = np.where(hits.any(axis=1), 1.0 / np.log2(ranks + 1), 0.0)
        result[f"N@{k}"] = float(gains.mean())
    return result
