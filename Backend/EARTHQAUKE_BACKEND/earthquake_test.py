"""
Standalone testing script for the Earthquake Alert Classification model.

After running the Jupyter notebook, this script expects:
earthquake_outputs/
    best_earthquake_alert_model.joblib
    earthquake_label_encoder.joblib
    earthquake_features.joblib

Run:
    python earthquake_standalone_test.py

Then enter each feature when prompted, or call predict_earthquake(...)
from another Python program.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "earthquake_outputs")

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_earthquake_alert_model.joblib")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "earthquake_label_encoder.joblib")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "earthquake_features.joblib")

FEATURES = joblib.load(FEATURES_PATH)
MODEL = joblib.load(MODEL_PATH)
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)


def predict_earthquake(magnitude, depth, cdi, mmi, sig):
    """
    Predict earthquake alert class using keyword inputs.

    Example:
        result = predict_earthquake(
            magnitude=6.5,
            depth=30,
            cdi=6,
            mmi=5,
            sig=100
        )
    """
    values = {
        "magnitude": float(magnitude),
        "depth": float(depth),
        "cdi": float(cdi),
        "mmi": float(mmi),
        "sig": float(sig)
    }

    # Keep exactly the feature order used during training.
    input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    encoded_prediction = MODEL.predict(input_df)[0]
    predicted_class = LABEL_ENCODER.inverse_transform([encoded_prediction])[0]

    confidence = None
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(input_df)[0]
        confidence = float(probabilities.max())

    return predicted_class, confidence


def main():
    print("=" * 60)
    print("EARTHQUAKE ALERT PREDICTION")
    print("=" * 60)
    print("Enter the earthquake measurements.")
    print()

    try:
        magnitude = float(input("Enter magnitude: "))
        depth = float(input("Enter depth: "))
        cdi = float(input("Enter cdi: "))
        mmi = float(input("Enter mmi: "))
        sig = float(input("Enter sig: "))

        result, confidence = predict_earthquake(
            magnitude=magnitude,
            depth=depth,
            cdi=cdi,
            mmi=mmi,
            sig=sig
        )

        print()
        print("-" * 60)
        print("PREDICTION RESULT")
        print("-" * 60)
        print("Magnitude :", magnitude)
        print("Depth     :", depth)
        print("CDI       :", cdi)
        print("MMI       :", mmi)
        print("SIG       :", sig)
        print("Predicted Alert:", result)

        if confidence is not None:
            print(f"Model Confidence: {confidence:.2%}")

        print("-" * 60)

    except ValueError:
        print("Invalid input. Please enter numeric values only.")
    except FileNotFoundError:
        print(
            "\\nModel files were not found. Run the Jupyter notebook first "
            "to train and save the model."
        )


if __name__ == "__main__":
    main()
