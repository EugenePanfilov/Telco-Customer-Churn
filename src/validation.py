from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit

def make_cv(y, groups: Optional[np.ndarray] = None, time_col: Optional[pd.Series] = None, cfg: Optional[dict] = None):
    v = cfg["validation"] if cfg is not None else {"kind": "stratified_kfold","n_splits": 5,"shuffle": True,"random_state": 42}
    kind = v.get("kind", "stratified_kfold").lower()
    n_splits = int(v.get("n_splits", 5))
    shuffle = bool(v.get("shuffle", True))
    random_state = int(v.get("random_state", 42))

    if kind == "time":
        if time_col is None:
            raise ValueError("time_col is required for time-based split")
        order = np.argsort(pd.to_datetime(time_col).values)
        tss = TimeSeriesSplit(n_splits=n_splits)
        class _Wrapper:
            def split(self, X=None, y=None, groups=None):
                for tr, va in tss.split(order):
                    yield order[tr], order[va]
        return _Wrapper()

    if kind == "group_kfold":
        if groups is None:
            raise ValueError("groups are required for GroupKFold")
        return GroupKFold(n_splits=n_splits)

    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
