import pandas as pd
from typing import Tuple, List, Optional

def read_dataset(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num = [c for c in X.columns if c not in cat]
    return num, cat

def maybe_parse_time(X: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col and time_col in X.columns:
        X = X.copy()
        X[time_col] = pd.to_datetime(X[time_col], errors="coerce")
    return X
