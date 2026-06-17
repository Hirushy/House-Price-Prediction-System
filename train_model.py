import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import json

def create_synthetic_data(num_samples=50000):
    print(f"Generating {num_samples} records of synthetic data...")
    np.random.seed(42)
    
    # Define location hierarchy
    locations = {
        "Western": {
            "Colombo": ["Colombo", "Dehiwala", "Nugegoda", "Maharagama", "Mount Lavinia"],
            "Gampaha": ["Negombo", "Gampaha", "Kelaniya", "Wattala"],
            "Kalutara": ["Panadura", "Kalutara", "Horana"]
        },
        "Central": {
            "Kandy": ["Kandy", "Peradeniya", "Katugastota"],
            "Matale": ["Matale", "Dambulla"]
        },
        "Southern": {
            "Galle": ["Galle", "Hikkaduwa", "Ambalangoda"],
            "Matara": ["Matara", "Weligama"]
        },
        "Northern": {
            "Jaffna": ["Jaffna", "Nallur", "Chavakachcheri"]
        }
    }
    
    provinces = list(locations.keys())
    
    data = []
    
    for _ in range(num_samples):
        province = np.random.choice(provinces, p=[0.5, 0.2, 0.2, 0.1])
        district = np.random.choice(list(locations[province].keys()))
        town = np.random.choice(locations[province][district])
        
        # Location specific parameters
        if district == "Colombo":
            dist_colombo = np.random.uniform(0.5, 20.0)
            lat = np.random.uniform(6.85, 6.98)
            lon = np.random.uniform(79.85, 79.95)
            loc_multiplier = 1.8
            income = np.random.uniform(150000, 500000)
            dev_index = np.random.uniform(0.8, 1.0)
        elif district == "Gampaha":
            dist_colombo = np.random.uniform(20.0, 45.0)
            lat = np.random.uniform(7.0, 7.2)
            lon = np.random.uniform(79.8, 80.1)
            loc_multiplier = 1.3
            income = np.random.uniform(80000, 250000)
            dev_index = np.random.uniform(0.6, 0.8)
        elif district == "Kalutara":
            dist_colombo = np.random.uniform(25.0, 60.0)
            lat = np.random.uniform(6.5, 6.8)
            lon = np.random.uniform(79.9, 80.2)
            loc_multiplier = 1.2
            income = np.random.uniform(70000, 200000)
            dev_index = np.random.uniform(0.5, 0.7)
        elif district == "Kandy":
            dist_colombo = np.random.uniform(90.0, 130.0)
            lat = np.random.uniform(7.2, 7.4)
            lon = np.random.uniform(80.5, 80.7)
            loc_multiplier = 1.1
            income = np.random.uniform(80000, 220000)
            dev_index = np.random.uniform(0.6, 0.8)
        elif district == "Galle":
            dist_colombo = np.random.uniform(100.0, 140.0)
            lat = np.random.uniform(6.0, 6.3)
            lon = np.random.uniform(80.1, 80.4)
            loc_multiplier = 1.1
            income = np.random.uniform(75000, 200000)
            dev_index = np.random.uniform(0.5, 0.7)
        else:
            dist_colombo = np.random.uniform(120.0, 400.0)
            lat = np.random.uniform(5.9, 9.8)
            lon = np.random.uniform(79.8, 81.8)
            loc_multiplier = 0.9
            income = np.random.uniform(50000, 150000)
            dev_index = np.random.uniform(0.3, 0.6)
            
        # Property details
        prop_type = np.random.choice(["House", "Apartment", "Villa", "Land"], p=[0.7, 0.2, 0.05, 0.05])
        
        if prop_type == "Land":
            bedrooms = 0
            bathrooms = 0
            area = 0
            age = 0
            furnished = "Unfurnished"
            prop_condition = "Fair"
        elif prop_type == "Apartment":
            bedrooms = np.random.randint(1, 5)
            bathrooms = np.random.randint(1, bedrooms + 2)
            area = np.random.randint(500, 3000)
            age = np.random.randint(0, 20)
            furnished = np.random.choice(["Furnished", "Semi-Furnished", "Unfurnished"])
            prop_condition = np.random.choice(["Excellent", "Good", "Fair"], p=[0.5, 0.4, 0.1])
        else:
            bedrooms = np.random.randint(2, 8)
            bathrooms = np.random.randint(1, bedrooms + 2)
            area = np.random.randint(800, 8000)
            age = np.random.randint(0, 50)
            furnished = np.random.choice(["Furnished", "Semi-Furnished", "Unfurnished"])
            prop_condition = np.random.choice(["Excellent", "Good", "Fair", "Poor"], p=[0.3, 0.5, 0.15, 0.05])
            
        land_size = np.random.uniform(5.0, 80.0) if prop_type != "Apartment" else 0.0
        
        # Amenities & Environment
        dist_hospital = np.random.uniform(0.5, 15.0)
        dist_school = np.random.uniform(0.2, 10.0)
        dist_highway = np.random.uniform(1.0, 30.0)
        
        supermarkets = np.random.randint(0, 10)
        banks = np.random.randint(0, 15)
        schools = np.random.randint(0, 8)
        hospitals = np.random.randint(0, 5)
        
        traffic = np.random.uniform(0.1, 1.0)
        flood_risk = np.random.uniform(0.1, 1.0)
        crime_rate = np.random.uniform(0.1, 1.0)
        
        solar = 1 if np.random.random() > 0.8 else 0
        road_type = np.random.choice(["Carpet", "Concrete", "Gravel"], p=[0.6, 0.3, 0.1])
        
        # Additional features for production-grade model
        dist_airport = np.random.uniform(2.0, 50.0)
        dist_railway = np.random.uniform(0.5, 15.0)
        dist_beach = np.random.uniform(0.1, 100.0)
        
        has_pool = 1 if prop_type in ["Villa", "House"] and np.random.random() > 0.8 else 0
        if prop_type == "Apartment":
            has_pool = 1 if np.random.random() > 0.5 else 0
            
        has_gym = 1 if prop_type == "Apartment" and np.random.random() > 0.5 else (1 if prop_type == "Villa" else 0)
        has_garden = 1 if prop_type in ["House", "Villa"] else 0
        
        security_system = np.random.choice(["None", "Basic", "Advanced"], p=[0.5, 0.3, 0.2])
        if prop_type == "Apartment":
            security_system = np.random.choice(["Basic", "Advanced"], p=[0.4, 0.6])
            
        parking_spaces = np.random.randint(0, 5)
        if prop_type == "Apartment":
            parking_spaces = np.random.randint(1, 3)
            
        water_supply = np.random.choice(["Municipal", "Well", "Both"], p=[0.7, 0.1, 0.2])
        electricity_supply = np.random.choice(["Grid", "Grid+Solar", "Off-grid"], p=[0.8, 0.19, 0.01])
        internet = np.random.choice(["Fiber", "4G", "ADSL", "None"], p=[0.5, 0.3, 0.15, 0.05])
        
        air_quality = np.random.uniform(20.0, 150.0)
        safety_score = np.random.uniform(3.0, 10.0)
        walkability_score = np.random.uniform(1.0, 10.0)
        
        property_tax_rate = np.random.uniform(0.5, 2.5)
        year_built = 2026 - age
        has_maid_room = 1 if bedrooms > 3 and np.random.random() > 0.4 else 0
        balconies = np.random.randint(0, 4) if prop_type in ["Apartment", "House", "Villa"] else 0
        ac_rooms = np.random.randint(0, bedrooms + 1)
        
        # Base price logic as per user request
        # area * 12000 + bedrooms * 500000 + bathrooms * 300000 - age * 25000 - dist_colombo * 150000 + income * 20
        base_price = (
            (area * 12000) + 
            (bedrooms * 500000) + 
            (bathrooms * 300000) +
            (land_size * 500000) - # Land size value
            (age * 25000) - 
            (dist_colombo * 150000) + 
            (income * 20) +
            (has_pool * 2000000) + 
            (has_gym * 500000) +
            (parking_spaces * 300000) +
            (has_maid_room * 400000) +
            (balconies * 200000) +
            (ac_rooms * 150000)
        )
        
        # Apply location multiplier and constraints to avoid negative prices
        base_price = max(base_price, 2000000) * loc_multiplier
        
        # Add value for amenities and condition
        amenity_bonus = (supermarkets + banks + schools + hospitals) * 50000
        solar_bonus = solar * 1000000
        condition_mult = {"Excellent": 1.2, "Good": 1.0, "Fair": 0.8, "Poor": 0.6}[prop_condition]
        
        # Deduct for risks
        risk_penalty = (flood_risk * 500000) + (crime_rate * 500000) + (traffic * 200000) + (air_quality * 2000)
        
        noise = np.random.normal(0, 1000000)
        
        final_price = (base_price + amenity_bonus + solar_bonus - risk_penalty) * condition_mult + noise
        final_price = max(1000000, final_price) # Minimum price 1M
        
        data.append({
            "Province": province,
            "District": district,
            "Town": town,
            "Property_Type": prop_type,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floor_Area_sqft": area,
            "Land_Size_Perches": round(land_size, 2),
            "Property_Age": age,
            "Furnished_Status": furnished,
            "Property_Condition": prop_condition,
            "Road_Type": road_type,
            "Distance_to_Colombo_km": round(dist_colombo, 2),
            "Latitude": round(lat, 4),
            "Longitude": round(lon, 4),
            "Distance_to_Hospital_km": round(dist_hospital, 2),
            "Distance_to_School_km": round(dist_school, 2),
            "Distance_to_Highway_km": round(dist_highway, 2),
            "Distance_to_Airport_km": round(dist_airport, 2),
            "Distance_to_Railway_km": round(dist_railway, 2),
            "Distance_to_Beach_km": round(dist_beach, 2),
            "Nearby_Supermarket_Count": supermarkets,
            "Nearby_Banks_Count": banks,
            "Nearby_Schools_Count": schools,
            "Nearby_Hospitals_Count": hospitals,
            "Traffic_Congestion_Index": round(traffic, 2),
            "Flood_Risk_Score": round(flood_risk, 2),
            "Crime_Rate_Score": round(crime_rate, 2),
            "Air_Quality_Index": round(air_quality, 2),
            "Neighborhood_Safety_Score": round(safety_score, 2),
            "Walkability_Score": round(walkability_score, 2),
            "Development_Index": round(dev_index, 2),
            "Average_Income_Area": round(income, 2),
            "Property_Tax_Rate": round(property_tax_rate, 2),
            "Year_Built": year_built,
            "Solar_Power_Availability": solar,
            "Has_Swimming_Pool": has_pool,
            "Has_Gym": has_gym,
            "Has_Garden": has_garden,
            "Has_Maid_Room": has_maid_room,
            "Security_System": security_system,
            "Parking_Spaces": parking_spaces,
            "Water_Supply": water_supply,
            "Electricity_Supply": electricity_supply,
            "Internet_Connectivity": internet,
            "Balcony_Count": balconies,
            "AC_Rooms": ac_rooms,
            "Price": round(final_price, 2)
        })
        
    df = pd.DataFrame(data)
    return df

def train_model():
    df = create_synthetic_data(50000)
    df.to_csv("housing.csv", index=False)
    print("Saved 50,000 records to housing.csv")
    
    # Separate Features and Target
    X = df.drop("Price", axis=1)
    y = df["Price"]
    
    categorical_features = [
        "Province", "District", "Town", "Property_Type", 
        "Furnished_Status", "Road_Type", "Property_Condition",
        "Security_System", "Water_Supply", "Electricity_Supply",
        "Internet_Connectivity"
    ]
    
    numerical_features = [
        "Bedrooms", "Bathrooms", "Floor_Area_sqft", "Land_Size_Perches",
        "Property_Age", "Distance_to_Colombo_km", "Latitude", "Longitude",
        "Distance_to_Hospital_km", "Distance_to_School_km", "Distance_to_Highway_km",
        "Distance_to_Airport_km", "Distance_to_Railway_km", "Distance_to_Beach_km",
        "Nearby_Supermarket_Count", "Nearby_Banks_Count", "Nearby_Schools_Count",
        "Nearby_Hospitals_Count", "Traffic_Congestion_Index", "Flood_Risk_Score",
        "Crime_Rate_Score", "Air_Quality_Index", "Neighborhood_Safety_Score",
        "Walkability_Score", "Development_Index", "Average_Income_Area", 
        "Property_Tax_Rate", "Year_Built", "Solar_Power_Availability",
        "Has_Swimming_Pool", "Has_Gym", "Has_Garden", "Has_Maid_Room",
        "Parking_Spaces", "Balcony_Count", "AC_Rooms"
    ]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create full pipeline with XGBoost
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Pipeline...")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation -> MAE: Rs. {mae:,.2f} | R²: {r2:.4f}")
    
    joblib.dump(model, "house_model.pkl")
    print("Model saved to house_model.pkl")
    
    # Extract feature importances
    # XGBoost internal importances correspond to the transformed columns
    xgb_model = model.named_steps['regressor']
    importances = xgb_model.feature_importances_
    
    # Get feature names from ColumnTransformer
    cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(encoded_cat_features)
    
    # Create a clean feature importance dictionary for the top 15 features
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Group one-hot encoded categorical importances back to their parent category for easier reading on the frontend
    consolidated_importances = {}
    for _, row in feature_importance_df.iterrows():
        feat = row['Feature']
        imp = row['Importance']
        
        # Check if it's a categorical feature (e.g., District_Colombo)
        is_cat = False
        for cat in categorical_features:
            if feat.startswith(cat + "_"):
                consolidated_importances[cat] = consolidated_importances.get(cat, 0) + imp
                is_cat = True
                break
        
        if not is_cat:
            consolidated_importances[feat] = imp
            
    # Sort and take top 15
    top_importances = sorted(consolidated_importances.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Save to JSON for the backend to serve
    importances_list = [{"name": k, "value": round(float(v), 4)} for k, v in top_importances]
    with open("feature_importances.json", "w") as f:
        json.dump(importances_list, f)
        
    print("Feature importances saved to feature_importances.json")

if __name__ == "__main__":
    train_model()
