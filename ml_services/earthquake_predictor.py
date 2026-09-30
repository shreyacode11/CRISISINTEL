from utils.timeutil import now_ist, utc_now
import os, joblib, pandas as pd
from config import Config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "earthquake_outputs")

_cache = {}

def _load():
    if not _cache:
        _cache["features"] = joblib.load(os.path.join(OUT, "earthquake_features.joblib"))
        _cache["model"]    = joblib.load(os.path.join(OUT, "best_earthquake_alert_model.joblib"))
        _cache["le"]       = joblib.load(os.path.join(OUT, "earthquake_label_encoder.joblib"))
    return _cache


def predict_earthquake(magnitude, depth, cdi, mmi, sig):
    c = _load()
    values = {
        "magnitude": float(magnitude),
        "depth": float(depth),
        "cdi": float(cdi),
        "mmi": float(mmi),
        "sig": float(sig),
    }
    df = pd.DataFrame([[values[f] for f in c["features"]]], columns=c["features"])
    enc = c["model"].predict(df)[0]
    label = c["le"].inverse_transform([enc])[0]
    conf = None
    if hasattr(c["model"], "predict_proba"):
        conf = float(c["model"].predict_proba(df)[0].max())
    return str(label), conf