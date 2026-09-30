from utils.timeutil import now_ist, utc_now
import os, joblib, numpy as np, pandas as pd
from config import Config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = None

def _load():
    global _pkg
    if _pkg is None:
        _pkg = joblib.load(os.path.join(BASE, "flood.pkl"))
    return _pkg


def predict_flood(inputs: dict):
    """inputs = {feature_name: value}"""
    pkg = _load()
    model = pkg["model"]
    scaler = pkg["scaler"]
    uses_scaler = pkg["uses_scaler"]
    feature_columns = pkg["feature_columns"]
    feature_encoders = pkg["feature_encoders"]
    target_encoder = pkg["target_encoder"]

    df = pd.DataFrame([inputs], columns=feature_columns)

    for feat, enc in feature_encoders.items():
        df[feat] = enc.transform(df[feat].astype(str))

    X = scaler.transform(df) if uses_scaler else df
    pred = model.predict(X)[0]

    if target_encoder is not None:
        result = target_encoder.inverse_transform([int(pred)])[0]
    else:
        result = pred

    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(X)[0]))

    return str(result), conf