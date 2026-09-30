
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "saved_models/best_cyclone_model.pkl"

with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

model = package["model"]
scaler = package["scaler"]
uses_scaled = package["uses_scaled"]
feature_columns = package["feature_columns"]
class_names = package["class_names"]
model_name = package["model_name"]

print("=" * 60)
print("CYCLONE PREDICTION - STANDALONE TESTING")
print("=" * 60)
print("Loaded model:", model_name)
print("Enter the environmental parameters below.")

sea_surface_temperature = float(input("Sea Surface Temperature: "))
atmospheric_pressure = float(input("Atmospheric Pressure: "))
humidity = float(input("Humidity: "))
wind_shear = float(input("Wind Shear: "))
vorticity = float(input("Vorticity: "))
latitude = float(input("Latitude: "))
ocean_depth = float(input("Ocean Depth: "))
proximity_to_coastline = float(input("Proximity to Coastline: "))
pre_existing_disturbance = int(input("Pre-existing Disturbance (0=No, 1=Yes): "))

input_data = pd.DataFrame([{
    "Sea_Surface_Temperature": sea_surface_temperature,
    "Atmospheric_Pressure": atmospheric_pressure,
    "Humidity": humidity,
    "Wind_Shear": wind_shear,
    "Vorticity": vorticity,
    "Latitude": latitude,
    "Ocean_Depth": ocean_depth,
    "Proximity_to_Coastline": proximity_to_coastline,
    "Pre_existing_Disturbance": pre_existing_disturbance
}])

input_data = input_data[feature_columns]

if uses_scaled:
    model_input = scaler.transform(input_data)
else:
    model_input = input_data

prediction = model.predict(model_input)[0]

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(model_input)[0]
    confidence = float(np.max(probability)) * 100
else:
    confidence = None

result = class_names.get(int(prediction), str(prediction))

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)
print("Prediction:", result)

if confidence is not None:
    print(f"Confidence: {confidence:.2f}%")

if int(prediction) == 1:
    print("Cyclone formation is predicted.")
else:
    print("No cyclone formation is predicted.")
print("=" * 60)
