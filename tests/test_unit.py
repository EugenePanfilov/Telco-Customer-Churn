# tests/test_unit.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ------------- validation.make_cv -------------
def test_cv_stratified_preserves_positive_rate():
    from src.validation import make_cv
    y = np.array([0, 1] * 50)  # общий positive rate = 0.5
    cfg = {"validation": {"kind": "stratified_kfold", "n_splits": 5, "shuffle": True, "random_state": 42}}
    cv = make_cv(y=y, groups=None, time_col=None, cfg=cfg)
    rates = []
    for tr, va in cv.split(np.zeros_like(y), y):
        rate = y[va].mean()
        rates.append(rate)
    # щедрый допуск "с запасом"
    assert all(abs(r - 0.5) < 0.11 for r in rates)

def test_cv_group_no_leak():
    from src.validation import make_cv
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    cfg = {"validation": {"kind": "group_kfold", "n_splits": 4}}
    cv = make_cv(y=y, groups=groups, time_col=None, cfg=cfg)
    for tr, va in cv.split(np.zeros_like(y), y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[va]))  # группа не делится между train/val

def test_cv_time_respects_order():
    from src.validation import make_cv
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    t = pd.Series(pd.date_range("2023-01-01", periods=len(y), freq="D"))
    cfg = {"validation": {"kind": "time", "n_splits": 3}}
    cv = make_cv(y=y, groups=None, time_col=t, cfg=cfg)
    last_train_max = -1
    for tr, va in cv.split(np.zeros_like(y)):
        assert tr.max() < va.min()          # train_time_max < val_time_min
        assert tr.max() > last_train_max    # строго вперёд
        last_train_max = tr.max()

# ------------- metrics.select_threshold / compute_metrics -------------
def test_metrics_keys_and_strategies():
    from src.metrics import select_threshold, compute_metrics
    y = np.array([0,0,1,1,0,1,0,1,0,1])
    p = np.array([0.1,0.2,0.7,0.8,0.4,0.9,0.3,0.6,0.2,0.8])

    # f_beta: ключи присутствуют + разумный порог
    sel_f = select_threshold(y_val=y, p_val=p, strategy="f_beta", fbeta=2.0)
    m = compute_metrics(y_true=y, p_pred=p, threshold=sel_f["threshold"])
    for k in ("roc_auc", "pr_auc", "logloss", "f1", "fbeta", "cm"):
        assert k in m

    # cost: минимизация стоимости (поддерживаем обе схемы ключей из реализации)
    sel_c = select_threshold(y_val=y, p_val=p, strategy="cost", cost_matrix={"fp": 1.0, "fn": 5.0})
    assert 0.0 <= sel_c["threshold"] <= 1.0
    assert ("value" in sel_c) or ("cost_at_threshold" in sel_c)

def test_basic_calibration_monotone_and_bounded():
    """
    Калибровка без deprecated cv='prefit':
    - вероятности в [0, 1]
    - высокая ранговая корреляция с базовым score (монотонность в среднем)
    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    base = LogisticRegression(max_iter=200).fit(X, y)
    p_raw = base.predict_proba(X)[:, 1]

    for method in ("sigmoid", "isotonic"):  # Platt / Isotonic
        cal = CalibratedClassifierCV(LogisticRegression(max_iter=200), method=method, cv=5).fit(X, y)
        p_cal = cal.predict_proba(X)[:, 1]
        # границы [0,1]
        assert (p_cal >= 0).all() and (p_cal <= 1).all()
        # высокая ранговая корреляция с исходным score
        rho, _ = spearmanr(p_raw, p_cal)
        assert rho > 0.98

# ------------- ColumnTransformer pipeline -------------
def test_features_pipeline_impute_onehot_handles_unknown():
    from src.features import build_preprocessor
    cfg = {
        "preprocess": {
            "num_cols": ["age"],
            "cat_cols": ["city"],
            "bin_cols": ["vip"],
            "numeric_imputer": "median",
            "scale_numeric": True,
            "cat_imputer": "most_frequent",
            "cat_encoding": "onehot",
        }
    }
    pre = build_preprocessor(cfg)

    X_train = pd.DataFrame({"age": [30.0, None, 50.0], "city": ["A", "B", "A"], "vip": [0, 1, 1]})
    Xt = pre.fit_transform(X_train)
    # нет NaN после импутации
    assert not pd.isna(Xt).to_numpy().any()

    # transform с новой категорией не падает (handle_unknown="ignore")
    X_new = pd.DataFrame({"age": [20.0], "city": ["C"], "vip": [0]})
    Xn = pre.transform(X_new)
    # размерность согласована
    assert Xt.shape[1] == Xn.shape[1]