from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             f1_score, fbeta_score, confusion_matrix, precision_recall_curve)

def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float) -> Dict:
    y_pred = (p_pred >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p_pred)),
        "pr_auc": float(average_precision_score(y_true, p_pred)),
        "logloss": float(log_loss(y_true, np.clip(p_pred, 1e-7, 1 - 1e-7))),
        "f1": float(f1_score(y_true, y_pred)),
        "fbeta": float(fbeta_score(y_true, y_pred, beta=2.0)),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }

def _maximize_fbeta(y: np.ndarray, p: np.ndarray, beta: float) -> Tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    best_t, best_s = 0.5, -1.0
    for t in np.r_[0.0, thr, 1.0]:
        s = fbeta_score(y, (p >= t).astype(int), beta=beta)
        if s > best_s:
            best_s, best_t = float(s), float(t)
    return best_t, best_s

def _minimize_cost(y: np.ndarray, p: np.ndarray, cost_fp: float, cost_fn: float) -> Tuple[float, float]:
    best_t, best_c = 0.5, float("inf")
    for t in np.linspace(0, 1, 2001):
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
        c = cost_fp * fp + cost_fn * fn
        if c < best_c:
            best_c, best_t = float(c), float(t)
    return best_t, best_c

def select_threshold(y_val: np.ndarray, p_val: np.ndarray, strategy: str = "f_beta",
                     fbeta: float = 2.0, cost_matrix: Optional[Dict[str, float]] = None) -> Dict:
    st = (strategy or "f_beta").lower()
    if st == "f_beta":
        t, s = _maximize_fbeta(y_val, p_val, beta=fbeta)
        return {"threshold": t, "objective": "fbeta", "value": s}
    if st == "cost":
        if not cost_matrix or "fp" not in cost_matrix or "fn" not in cost_matrix:
            raise ValueError("cost_matrix must have 'fp' and 'fn'")
        t, c = _minimize_cost(y_val, p_val, cost_matrix["fp"], cost_matrix["fn"])
        return {"threshold": t, "objective": "cost", "value": c}
    raise ValueError("Unknown threshold selection strategy")
