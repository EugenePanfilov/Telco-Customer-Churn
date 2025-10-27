# src/train.py
# 1) читаем конфиг -> 2) строим ColumnTransformer -> 3) KFold OOF -> 4) калибровка -> 5) save artifacts

import argparse
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from .data import read_dataset, maybe_parse_time
from .features import build_preprocessor
from .validation import make_cv
from .metrics import compute_metrics, select_threshold


def _build_model(cfg):
    name = cfg["model"]["name"].lower()
    params = dict(cfg["model"].get("params", {}))
    if name == "logreg":
        return LogisticRegression(
            max_iter=params.get("max_iter", 200),
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if name == "catboost":
        from catboost import CatBoostClassifier
        params.setdefault("verbose", False)
        return CatBoostClassifier(**params)
    raise ValueError(f"Unknown model name: {name}")


def main():
    # === 1) читаем конфиг ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    artifacts = cfg.get("artifacts_dir", "artifacts")
    os.makedirs(artifacts, exist_ok=True)

    # данные
    X, y = read_dataset(cfg["data"]["path_train"], cfg["data"]["target"])
    X = maybe_parse_time(X, cfg["data"].get("time_col"))
    groups = X[cfg["data"]["group_col"]].values if cfg["data"].get("group_col") in X.columns else None
    time_col = X[cfg["data"]["time_col"]] if cfg["data"].get("time_col") in X.columns else None

    # === 2) строим ColumnTransformer ===
    preprocessor = build_preprocessor(cfg)
    base_model = _build_model(cfg)

    # === 3) KFold OOF ===
    cv = make_cv(y=y.values, groups=groups, time_col=time_col, cfg=cfg)
    oof_raw = np.zeros(len(X), dtype=float)

    method = (cfg["metrics"].get("calibration", "platt") or "none").lower()
    has_cal = method in ("platt", "isotonic")
    oof_cal = np.zeros(len(X), dtype=float) if has_cal else None

    for tr_idx, va_idx in cv.split(X, y, groups):
        # на каждый фолд — свежие клоны препроцессора и модели
        pipe = Pipeline([("pre", clone(preprocessor)), ("model", clone(base_model))])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        p_val = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        oof_raw[va_idx] = p_val

        # === 4) калибровка (на валидации, для OOF-оценки) ===
        if has_cal:
            cal = CalibratedClassifierCV(
                pipe,
                method=("sigmoid" if method == "platt" else "isotonic"),
                cv="prefit",
            )
            cal.fit(X.iloc[va_idx], y.iloc[va_idx])
            oof_cal[va_idx] = cal.predict_proba(X.iloc[va_idx])[:, 1]

    use_pred = oof_cal if has_cal else oof_raw

    # выбор порога
    sel = select_threshold(
        y_val=y.values,
        p_val=use_pred,
        strategy=cfg["metrics"].get("threshold_strategy", "f_beta"),
        fbeta=float(cfg["metrics"].get("fbeta", 2.0)),
        cost_matrix=cfg["metrics"].get("cost_matrix"),
    )
    threshold = float(sel["threshold"])

    # метрики по OOF
    metrics_raw = compute_metrics(y.values, oof_raw, threshold)
    metrics_cal = compute_metrics(y.values, use_pred, threshold) if has_cal else None

    # финальная модель (с калибровкой при необходимости)
    final_pipe = Pipeline([("pre", clone(preprocessor)), ("model", clone(base_model))])
    final_pipe.fit(X, y)
    if has_cal:
        final_pipe = CalibratedClassifierCV(
            final_pipe,
            method=("sigmoid" if method == "platt" else "isotonic"),
            cv=5,
        )
        final_pipe.fit(X, y)

    # калибровочный график
    plt.figure(figsize=(6, 5), dpi=140)
    frac_raw, mean_raw = calibration_curve(y.values, oof_raw, n_bins=10, strategy="quantile")
    plt.plot(mean_raw, frac_raw, "o-", label="Raw")
    if has_cal:
        frac_cal, mean_cal = calibration_curve(y.values, use_pred, n_bins=10, strategy="quantile")
        plt.plot(mean_cal, frac_cal, "o-", label="Calibrated")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts, "calibration_plot.png"))
    plt.close()

    # === 5) save artifacts ===
    joblib.dump(
        {"model": final_pipe, "threshold": threshold, "cfg": cfg},
        os.path.join(artifacts, "model.pkl"),
    )
    with open(os.path.join(artifacts, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection": sel,
                "oof_raw": metrics_raw,
                "oof_calibrated": metrics_cal,
                "used_calibration": (method if has_cal else "none"),
                "threshold": threshold,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] artifacts saved to {artifacts}")


if __name__ == "__main__":
    main()