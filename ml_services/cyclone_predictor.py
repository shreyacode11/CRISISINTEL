from utils.timeutil import now_ist, utc_now
import os
import pickle
import numpy as np
import pandas as pd
from config import Config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(BASE, "saved_models", "best_cyclone_model.pkl")
_cache = {}


def _load():
    if not _cache:
        with open(PKG, "rb") as f:
            _cache.update(pickle.load(f))
    return _cache


def predict_cyclone(params: dict):
    c = _load()

    df = pd.DataFrame([params])[c["feature_columns"]]
    X = c["scaler"].transform(df) if c["uses_scaled"] else df

    pred = c["model"].predict(X)[0]

    conf = None

    if hasattr(c["model"], "predict_proba"):
        raw = float(np.max(c["model"].predict_proba(X)[0]))

        # Normalize: if the value is > 1, the model already returns percentages
        if raw > 1.0:
            conf = raw
        else:
            conf = raw * 100

        # Clamp confidence to the valid range [0, 100]
        conf = round(max(0.0, min(100.0, conf)), 2)

    label = c["class_names"].get(int(pred), str(pred))

    return label, conf