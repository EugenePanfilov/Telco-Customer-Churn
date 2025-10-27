# src/features.py
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """
    Колонки берём ТОЛЬКО из cfg['preprocess'].
      - num_cols: SimpleImputer -> (опц.) StandardScaler
      - cat_cols: SimpleImputer -> OneHotEncoder(handle_unknown='ignore')
      - bin_cols: SimpleImputer -> OneHotEncoder(drop='if_binary')  # 0/1
    """
    if not cfg or "preprocess" not in cfg:
        raise ValueError("Config with 'preprocess' section is required")

    p = cfg["preprocess"]
    num_cols: List[str] = p.get("num_cols", []) or []
    cat_cols: List[str] = p.get("cat_cols", []) or []
    bin_cols: List[str] = p.get("bin_cols", []) or []
    if not (num_cols or cat_cols or bin_cols):
        raise ValueError("Укажи хотя бы один список: num_cols / cat_cols / bin_cols")

    # numeric
    num_steps = [("imputer", SimpleImputer(strategy=p.get("numeric_imputer", "median")))]
    if p.get("scale_numeric", True):
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # categorical
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=p.get("cat_imputer", "most_frequent"), fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # binary -> one-hot with drop='if_binary' => один столбец 0/1
    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=p.get("cat_imputer", "most_frequent"), fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if bin_cols:
        transformers.append(("bin", bin_pipe, bin_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)