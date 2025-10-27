# ── quiet mode & safe backend ────────────────────────────────────────────────
import os, warnings
os.environ["MPLBACKEND"] = "Agg"  # безопасный бэкенд без GUI

from sklearn import set_config
set_config(transform_output="pandas")  # чтобы трансформеры возвращали DataFrame

from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning, FitFailedWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["MPLBACKEND"] = "Agg"   # безопасный headless-бэкенд
import argparse, json, os, joblib, numpy as np
from .data import read_dataset, maybe_parse_time
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, log_loss, confusion_matrix

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    import yaml; cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    bundle = joblib.load(os.path.join(cfg.get("artifacts_dir","artifacts"), "model.pkl"))
    model, thr = bundle["model"], float(bundle.get("threshold", 0.5))

    path = cfg["data"].get("path_test") or cfg["data"]["path_train"]
    X, y = read_dataset(path, cfg["data"]["target"]); X = maybe_parse_time(X, cfg["data"].get("time_col"))
    p = model.predict_proba(X)[:,1]; y_pred = (p >= thr).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "threshold_used": thr,
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "data_path": path,
    }
    out = os.path.join(cfg.get("artifacts_dir","artifacts"), "metrics_eval.json")
    with open(out, "w", encoding="utf-8") as f: json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[OK] Evaluation metrics saved to", out)

if __name__ == "__main__":
    main()
