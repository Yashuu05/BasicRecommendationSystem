import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.utils.model_utils import load_model
import pandas as pd
import joblib

# Correct model path
model_path = os.path.join(os.path.dirname(__file__), 'models', 'PricePrediction', 'best_model.pkl')

price_input = {
        "issue" : "broken_screen",
        "device" : 'laptop',
        "severity" : "medium",
        "urgent" : "no",
        "brand" : "HP",
        "device_age_years" : 4,
        "service_type" : "home_service",
        "warranty_status" : "out_of_warranty",
        "city_tier" : "tier1",
        "technician_experience" : 10
    }

# Load the model
model = joblib.load(filename=model_path)
# Convert input to DataFrame and predict
input_df = pd.DataFrame([price_input])
predicted_price = model.predict(input_df)
print(f"Predicted price: {predicted_price[0]}")

