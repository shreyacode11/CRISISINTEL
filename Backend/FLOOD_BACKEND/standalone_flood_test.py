import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "flood.pkl"
package = joblib.load(MODEL_PATH)
model = package["model"]
scaler = package["scaler"]
uses_scaler = package["uses_scaler"]
feature_columns = package["feature_columns"]
feature_encoders = package["feature_encoders"]
target_encoder = package["target_encoder"]

print("\n===== FLOOD PREDICTION SYSTEM =====")
values = {}

for feature in feature_columns:
    while True:
        try:
            raw = input(f"Enter {feature}: ")
            if feature in feature_encoders:
                values[feature] = raw
            else:
                values[feature] = float(raw)
            break
        except ValueError:
            print("Invalid input. Please enter a valid value.")

input_df = pd.DataFrame([values], columns=feature_columns)

for feature, encoder in feature_encoders.items():
    try:
        input_df[feature] = encoder.transform(input_df[feature].astype(str))
    except ValueError:
        print(f"Unknown value for {feature}. Allowed values: {list(encoder.classes_)}")
        raise SystemExit

if uses_scaler:
    input_data = scaler.transform(input_df)
else:
    input_data = input_df

prediction = model.predict(input_data)[0]

if target_encoder is not None:
    result = target_encoder.inverse_transform([int(prediction)])[0]
else:
    result = prediction

print("\n===================================")
print("        FLOOD PREDICTION RESULT")
print("===================================")
print("Prediction:", result)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_data)[0]
    print(f"Confidence: {np.max(probabilities) * 100:.2f}%")

print("===================================\n")
