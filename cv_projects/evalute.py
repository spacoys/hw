
import torch
import numpy as np
from sklearn.metrics import pairwise_distances

def compute_fnrm_at_fmr(embeddings, labels, fmr_target=0.0001):
    """
    Вычисляет FNMR при заданном FMR.
    
    FNMR (False Non-Match Rate) — доля правильно сопоставленных пар,
    которые модель ошибочно классифицировала как разные.
    FMR (False Match Rate) — доля неправильных пар,
    которые модель ошибочно классифицировала как одинаковые.
    
    Args:
        embeddings: [N, D] тензор эмбеддингов
        labels: [N] тензор меток
        fmr_target: целевой FMR (например, 0.0001)
    
    Returns:
        FNMR при заданном FMR
    """
    embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
    labels = labels.numpy() if torch.is_tensor(labels) else labels

    dists = pairwise_distances(embeddings, metric='cosine')
    N = len(labels)
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    pair_dists = [dists[i, j] for i, j in pairs]
    pair_labels = [1 if labels[i] == labels[j] else 0 for i, j in pairs]  # 1 = genuine, 0 = impostor
    
    pair_dists = np.array(pair_dists)
    pair_labels = np.array(pair_labels)
    
    genuine_dists = pair_dists[pair_labels == 1]
    impostor_dists = pair_dists[pair_labels == 0]

    sorted_impostor = np.sort(impostor_dists)
    threshold_idx = int(len(sorted_impostor) * fmr_target)
    threshold = sorted_impostor[threshold_idx] if threshold_idx < len(sorted_impostor) else sorted_impostor[-1]

    fnrm = np.mean(genuine_dists > threshold)
    
    return fnrm


def compute_metrics(embeddings, labels, thresholds=None):
    """Вычисляет полный набор метрик для разных порогов"""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
    labels = labels.numpy() if torch.is_tensor(labels) else labels
    
    dists = pairwise_distances(embeddings, metric='cosine')
    N = len(labels)
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    pair_dists = np.array([dists[i, j] for i, j in pairs])
    pair_labels = np.array([1 if labels[i] == labels[j] else 0 for i, j in pairs])
    
    genuine_dists = pair_dists[pair_labels == 1]
    impostor_dists = pair_dists[pair_labels == 0]
    
    results = []
    for thresh in thresholds:
        fmr = np.mean(impostor_dists <= thresh)
        fnmr = np.mean(genuine_dists > thresh)
        results.append({'threshold': thresh, 'FMR': fmr, 'FNMR': fnmr})
    
    return results