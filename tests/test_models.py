"""Automated tests for the CrisisIntel ML models (flood, cyclone, earthquake).

Run from the project root:   pytest tests/ -v
These replace the interactive input() test scripts so a CI server (Jenkins)
can run them without a keyboard. Each test passes/fails automatically.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml_services.flood_predictor import predict_flood            # noqa: E402
from ml_services.earthquake_predictor import predict_earthquake  # noqa: E402
from ml_services.cyclone_predictor import predict_cyclone        # noqa: E402

# ---------- sample inputs ----------
CYCLONE_YES = {   # first row of cyclone_dataset.csv (label: Cyclone)
    "Sea_Surface_Temperature": 27.4981604754, "Atmospheric_Pressure": 1008.5214291923,
    "Humidity": 89.2797576725, "Wind_Shear": 13.979877263, "Vorticity": 1.98e-05,
    "Latitude": 8.1198904067, "Ocean_Depth": 76.1376254757,
    "Proximity_to_Coastline": 1.3661761458, "Pre_existing_Disturbance": 1,
}
FLOOD_SAMPLE = {  # first row of flood_processed.csv
    "X": 3.90944444444444, "Y": 7.44305555555555, "Slope": 46.6861419677734,
    "Curvature ": -3888000000.0, "Aspect": 45.0, "TWI": -3.25036787986755,
    "FA": 147.0, "Drainage": 228.8528, "Rainfall": 101.51561643835623,
}


# ---------- cyclone ----------
def test_cyclone_returns_valid_label():
    label, _ = predict_cyclone(CYCLONE_YES)
    assert label in ("Cyclone", "No Cyclone")


def test_cyclone_missing_feature_raises():
    bad = dict(CYCLONE_YES)
    bad.pop("Humidity")
    with pytest.raises(KeyError):
        predict_cyclone(bad)


# ---------- earthquake ----------
def test_earthquake_returns_valid_alert_level():
    label, conf = predict_earthquake(magnitude=6.5, depth=30, cdi=7, mmi=6, sig=800)
    assert label in ("green", "yellow", "orange", "red")
    assert conf is None or 0.0 <= conf <= 1.0


def test_earthquake_strong_event_not_green():
    label, _ = predict_earthquake(magnitude=8.0, depth=10, cdi=9, mmi=9, sig=1500)
    assert label != "green"


# ---------- flood ----------
def test_flood_returns_susceptibility_class():
    label, conf = predict_flood(FLOOD_SAMPLE)
    assert str(label) in {"No_Flood", "Low", "Moderate", "High", "Very_High"}
    assert conf is None or 0.0 <= conf <= 1.0
