from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="House Price Prediction API")

# Setup CORS to allow React frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "house_model.pkl"
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        model = None
        print("Model not found. Please run train_model.py first.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

import pandas as pd
import json

class HouseFeatures(BaseModel):
    # Basic Features
    Bedrooms: int
    Bathrooms: int
    Floor_Area_sqft: int
    Location_Score: int = 8 # Used as a proxy to derive some location defaults
    
    # Advanced Features - Defaults provided so they are optional for "Basic" mode
    Province: str = "Western"
    District: str = "Colombo"
    Town: str = "Colombo"
    Property_Type: str = "House"
    Furnished_Status: str = "Unfurnished"
    Road_Type: str = "Carpet"
    Property_Condition: str = "Good"
    Security_System: str = "Basic"
    Water_Supply: str = "Municipal"
    Electricity_Supply: str = "Grid"
    Internet_Connectivity: str = "Fiber"
    
    Land_Size_Perches: float = 10.0
    Property_Age: int = 5
    Distance_to_Colombo_km: float = 5.0
    Latitude: float = 6.92
    Longitude: float = 79.86
    Distance_to_Hospital_km: float = 2.0
    Distance_to_School_km: float = 1.5
    Distance_to_Highway_km: float = 5.0
    Distance_to_Airport_km: float = 30.0
    Distance_to_Railway_km: float = 3.0
    Distance_to_Beach_km: float = 2.0
    
    Nearby_Supermarket_Count: int = 3
    Nearby_Banks_Count: int = 4
    Nearby_Schools_Count: int = 2
    Nearby_Hospitals_Count: int = 1
    Traffic_Congestion_Index: float = 0.5
    Flood_Risk_Score: float = 0.1
    Crime_Rate_Score: float = 0.1
    Air_Quality_Index: float = 50.0
    Neighborhood_Safety_Score: float = 8.0
    Walkability_Score: float = 7.0
    Development_Index: float = 0.8
    Average_Income_Area: float = 150000.0
    Property_Tax_Rate: float = 1.0
    Year_Built: int = 2021
    Solar_Power_Availability: int = 0
    Has_Swimming_Pool: int = 0
    Has_Gym: int = 0
    Has_Garden: int = 1
    Has_Maid_Room: int = 0
    Parking_Spaces: int = 2
    Balcony_Count: int = 1
    AC_Rooms: int = 1

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")
    
    # Convert input to DataFrame
    input_dict = features.dict()
    # Remove the mock location score as it's not in the model
    if 'Location_Score' in input_dict:
        del input_dict['Location_Score']
        
    input_df = pd.DataFrame([input_dict])
    
    try:
        prediction = model.predict(input_df)[0]
        
        # Simple heuristic confidence score: High if reasonable inputs, slightly lower if extremes
        confidence = 95
        if features.Floor_Area_sqft > 5000 or features.Bedrooms > 6:
            confidence -= 10
        if features.Property_Age > 30:
            confidence -= 5
            
        return {
            "estimated_price": round(float(prediction), 2),
            "currency": "LKR",
            "confidence_score": confidence
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/feature-importance")
def get_feature_importance():
    try:
        with open("feature_importances.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Feature importance data not found.")
