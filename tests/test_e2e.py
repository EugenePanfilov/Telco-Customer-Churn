# tests/test_e2e.py
import os, sys, runpy, json
from pathlib import Path
import numpy as np, pandas as pd
import yaml

def _run(entry, args):
    old = sys.argv[:]
    try:
        sys.argv = [entry] + args
        runpy.run_module(entry, run_name="__main__")
    finally:
        sys.argv = old

def _tiny_df(n=120, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n).round(1),
        "income": rng.normal(50_000, 12_000, n).round(0),
        "city": rng.choice(["A","B","C"], n),
        "segment": rng.choice(["retail","biz"], n),
        "is_vip": rng.integers(0, 2, n),
        "email_confirmed": rng.integers(0, 2, n),
    })
    p = 1/(1+np.exp(-(df["is_vip"]*1.1 + (df["income"]>52_000).astype(int)*0.8 - 0.7)))
    df["Churn"] = (rng.random(n) < p).astype(int)
    return df

def test_train_and_evaluate_end_to_end(tmp_path):
    # --- данные и конфиг ---
    data_dir = tmp_path / "data"; data_dir.mkdir()
    train_csv = data_dir / "train.csv"
    _tiny_df().to_csv(train_csv, index=False)

    art_dir = tmp_path / "artifacts"; art_dir.mkdir()
    cfg = {
        "data": {"path_train": str(train_csv), "path_test": None, "target": "Churn",
                 "time_col": None, "group_col": None},
        "validation": {"kind": "stratified_kfold", "n_splits": 3, "shuffle": True, "random_state": 42},
        "metrics": {"primary": "pr_auc", "secondary": ["roc_auc","logloss"],
                    "threshold_strategy": "f_beta", "fbeta": 2.0,
                    "cost_matrix": {"fp":1.0,"fn":5.0}, "calibration": "platt"},
        "model": {"name": "logreg", "params": {"max_iter": 200}},
        "preprocess": {
            "num_cols": ["age","income"],
            "cat_cols": ["city","segment"],
            "bin_cols": ["is_vip","email_confirmed"],
            "numeric_imputer": "median", "scale_numeric": True,
            "cat_imputer": "most_frequent", "cat_encoding": "onehot"},
        "seed": 42, "artifacts_dir": str(art_dir),
    }
    cfg_dir = tmp_path / "configs"; cfg_dir.mkdir()
    cfg_path = cfg_dir / "config_min.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # --- train → артефакты и версионирование ---
    _run("src.train", ["--config", str(cfg_path)])
    runs_dir = art_dir / "runs"
    assert runs_dir.exists(), "artifacts/runs/ не создан"
    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    assert run_folders, "нет ни одного запуска в runs/"
    last = run_folders[-1]
    for fn in ("model.pkl", "metrics.json", "calibration_plot.png"):
        assert (last / fn).exists(), f"{fn} отсутствует в {last}"

    # Проверим структуру metrics.json (ужатая: before/after)
    data = json.loads((last / "metrics.json").read_text(encoding="utf-8"))
    assert "before" in data
    for sect in ["before"] + (["after"] if "after" in data else []):
        for k in ("roc_auc","pr_auc","logloss","f1","fbeta","cm","threshold","calibration"):
            assert k in data[sect]

    # --- evaluate → пишет метрики рядом с моделью (latest) ---
    _run("src.evaluate", ["--config", str(cfg_path)])
    latest = art_dir / "latest"
    base = Path(os.path.realpath(latest)) if latest.exists() else last
    out = base / "metrics_eval.json"
    assert out.exists(), "metrics_eval.json не записан"

    e = json.loads(out.read_text(encoding="utf-8"))
    for k in ("roc_auc","pr_auc","logloss","f1","fbeta","cm","threshold","calibration"):
        assert k in e

def test_random_model_sanity():
    # Санити: случайные предсказания → ROC-AUC≈0.5, PR-AUC≈base_rate
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.random(1000)
    base_rate = y.mean()
    auc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    assert abs(auc - 0.5) < 0.05
    assert abs(pr - base_rate) < 0.05