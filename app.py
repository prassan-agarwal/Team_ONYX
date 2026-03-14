import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('house_price_xgb_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, expected_features = load_models()

st.title("🏡 House Price Prediction AI")
st.markdown("Predict the market value of properties in India using a highly accurate XGBoost Machine Learning Regression model trained on 12k properties with expanded features.")

col1, col2 = st.columns(2)

with st.sidebar:
    st.header("House Configuration")
    city = st.selectbox('City', ['Mumbai', 'Bangalore', 'Pune', 'Hyderabad', 'Nagpur', 'Kolkata'], index=0)
    locality_tier = st.selectbox('Locality Tier', ['Tier 1', 'Tier 2', 'Tier 3'], index=0)
    bhk = st.slider('BHK (Bedrooms)', 1, 5, 2)
    bathrooms = st.slider('Bathrooms', 1, 6, 2)
    balcony_count = st.slider('Balcony Count', 0, 5, 1)

    super_area = st.number_input('Super Area (sqft)', 300, 5000, 1000)
    carpet_area = st.number_input('Carpet Area (sqft)', 200, 4500, 800)
    floor_no = st.slider('Floor Number', 0, 40, 5)
    total_floors = st.slider('Total Floors in Building', 1, 60, 10)
    property_age = st.slider('Property Age (Years)', 0, 50, 5)

with col1:
    st.header("Proximity & Infrastructure")
    dist_metro = st.number_input('Distance to Metro (km)', 0.0, 20.0, 2.0)
    dist_city = st.number_input('Distance to City Center (km)', 0.0, 50.0, 10.0)
    dist_school = st.number_input('Nearby School (km)', 0.0, 20.0, 2.0)
    dist_hospital = st.number_input('Nearby Hospital (km)', 0.0, 20.0, 2.0)
    dist_it_hub = st.number_input('Distance to IT Hub (km)', 0.0, 30.0, 5.0)
    road_width = st.number_input('Road Width (ft)', 10, 100, 30)

    crime_rate = st.number_input('Crime Rate Index (0 = Safest)', 0, 100, 30)

with col2:
    st.header("Amenities & Property Details")
    furnishing = st.selectbox('Furnishing', ['Unfurnished', 'Semi-Furnished', 'Fully Furnished'])
    facing = st.selectbox('Facing', ["North", "South-West", "North-East", "South", "East", "North-West", "South-East", "West"])
    property_type = st.selectbox('Property Type', ["Independent House", "Apartment", "Villa", "Builder Floor"])
    floor_type = st.selectbox('Floor Type', ["Vitrified", "Wooden", "Marble", "Ceramic", "Granite"])
    builder_tier = st.selectbox('Builder Tier', ["Local", "Branded", "Mid-Tier"])
    ownership_type = st.selectbox('Ownership Type', ["Leasehold", "Freehold"])
    transaction_type = st.selectbox('Transaction Type', ["Resale", "New"])

    st.subheader("Ratings & Approvals")
    water_rating = st.slider('Water Supply Rating', 0.0, 10.0, 7.0, 0.1)
    power_rating = st.slider('Power Supply Rating', 0.0, 10.0, 7.0, 0.1)
    
    col_a, col_b = st.columns(2)
    with col_a:
        parking = st.radio("Parking Available?", ["Yes", "No"])
        lift = st.radio("Lift Available?", ["Yes", "No"])
    with col_b:
        gated = st.radio("Gated Society?", ["Yes", "No"])
        rera = st.radio("RERA Registered?", ["Yes", "No"])
        bank_app = st.radio("Bank Approved?", ["Yes", "No"])

def predict_price():
    # Convert Yes/No to 1/0
    p_map = {"Yes": 1, "No": 0}
    
    # Feature engineering logic from notebook
    area_ratio = carpet_area / super_area if super_area > 0 else 0
    total_prox = dist_metro + dist_city + dist_school + dist_hospital + dist_it_hub
    amenity_score = p_map[parking] + p_map[lift] + p_map[gated]
    floor_ratio = floor_no / (total_floors + 1)
    
    if bhk < 2: price_seg = "Small"
    elif bhk == 2: price_seg = "Medium"
    else: price_seg = "Large"

    # Base dictionary
    input_dict = {
        'BHK': bhk,
        'Bathrooms': bathrooms,
        'Super_Area_sqft': super_area,
        'Carpet_Area_sqft': carpet_area,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Property_Age_years': property_age,
        'Parking': p_map[parking],
        'Lift': p_map[lift],
        'Gated_Society': p_map[gated],
        'Distance_to_Metro_km': dist_metro,
        'Distance_to_CityCenter_km': dist_city,
        'Nearby_School_km': dist_school,
        'Nearby_Hospital_km': dist_hospital,
        'Crime_Rate_Index': crime_rate,
        # New Numeric features
        'Water_Supply_Rating': water_rating,
        'Power_Supply_Rating': power_rating,
        'RERA_Registered': p_map[rera],
        'Balcony_Count': balcony_count,
        'Road_Width_ft': road_width,
        'Bank_Approved': p_map[bank_app],
        'Distance_to_IT_Hub_km': dist_it_hub,
        # Engineered features
        'Area_Ratio': area_ratio,
        'Total_Proximity_Score': total_prox,
        'Amenity_Score': amenity_score,
        'Floor_Ratio': floor_ratio
    }

    df_input = pd.DataFrame([input_dict])

    # Ensure all expected columns are present
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Fill in the One-hot encoded matches
    fields = [
        ("City_", city),
        ("Locality_Tier_", locality_tier),
        ("Furnishing_", furnishing),
        ("Price_Segment_", price_seg),
        ("Facing_", facing),
        ("Builder_Tier_", builder_tier),
        ("Ownership_Type_", ownership_type),
        ("Transaction_Type_", transaction_type),
        ("Property_Type_", property_type),
        ("Floor_Type_", floor_type)
    ]
    
    for prefix, value in fields:
        col_name = f"{prefix}{value}"
        if col_name in expected_features:
            df_input[col_name] = 1

    df_input = df_input[expected_features]
    log_pred = model.predict(df_input)[0]
    return np.expm1(log_pred)

def apply_business_rules(base_price):
    price = base_price
    breakdown = []
    
    # 1. Layout & Interior Logic
    area_ratio = carpet_area / super_area if super_area > 0 else 0
    if bathrooms < bhk:
        price *= 0.90
        breakdown.append("-10% (Fewer Bathrooms than Bedrooms)")
    elif bathrooms > bhk:
        price *= 1.05
        breakdown.append("+5% (Luxury Bathroom Count)")
    if area_ratio > 0.8:
        price *= 1.06
        breakdown.append("+6% (Highly Efficient Carpet Area)")
    elif area_ratio < 0.6:
        price *= 0.90
        breakdown.append("-10% (Poor Layout / Wasted Super Area)")
    if super_area > 2000:
        price *= 1.08
        breakdown.append("+8% (Large Luxury Living Space)")

    # 2. Floor Related Logic
    if floor_no == 0:
        price *= 0.92
        breakdown.append("-8% (Ground Floor Privacy/Noise Concerns)")
    elif total_floors > 0 and floor_no == total_floors:
        price *= 0.95
        breakdown.append("-5% (Top Floor Heat Penalty)")
    elif total_floors > 0 and abs(floor_no - (total_floors / 2)) <= 2:
        price *= 1.04
        breakdown.append("+4% (Optimal Mid-Floor Placement)")
    if floor_no > 25:
        price *= 0.96
        breakdown.append("-4% (Very High Floor / Lift Risk)")

    # 3. Vastu / Facing Logic
    if facing == "East": 
        price *= 1.05
        breakdown.append("+5% (East Facing Vastu Premium)")
    elif facing == "North": 
        price *= 1.04
        breakdown.append("+4% (North Facing Vastu Premium)")
    elif facing == "North-East": 
        price *= 1.07
        breakdown.append("+7% (North-East Facing Vastu Premium)")
    elif facing == "South-West": 
        price *= 0.96
        breakdown.append("-4% (South-West Facing Penalty)")

    # 4. Parking Rules
    if parking == "No": 
        price *= 0.90
        breakdown.append("-10% (No Parking Available)")
    elif parking == "Yes": 
        price *= 1.03
        breakdown.append("+3% (Parking Available)")
        if super_area > 1800: 
            price *= 1.04
            breakdown.append("+4% (Luxury Property Parking Premium)")

    # 5. Builder Quality Logic
    if builder_tier == "Branded": 
        price *= 1.10
        breakdown.append("+10% (Branded/Reputed Builder)")
    elif builder_tier == "Local": 
        price *= 0.95
        breakdown.append("-5% (Local Builder Risk)")

    # 6. Age Logic
    if property_age <= 2: 
        price *= 1.05
        breakdown.append("+5% (New/Recently Built Property)")
    elif property_age > 35: 
        price *= 0.80
        breakdown.append("-20% (Very Old Building Depreciation)")
    elif property_age > 20: 
        price *= 0.90
        breakdown.append("-10% (Old Property Depreciation)")

    # 7. Balcony Logic
    if balcony_count >= 2: 
        price *= 1.03
        breakdown.append("+3% (Multiple Balconies)")
    if balcony_count > bhk + 2: 
        price *= 0.95
        breakdown.append("-5% (Excess Balcony / Wasted Space)")

    # 8. Infrastructure Logic
    if dist_metro < 1: 
        price *= 1.10
        breakdown.append("+10% (Walking Distance to Metro)")
    elif dist_metro < 3: 
        price *= 1.05
        breakdown.append("+5% (Near Metro Station)")
    elif dist_metro > 10: 
        price *= 0.95
        breakdown.append("-5% (Far from Metro Access)")
    
    if dist_it_hub < 5: 
        price *= 1.06
        breakdown.append("+6% (Close to IT Hub/Employment)")
    elif dist_it_hub > 20: 
        price *= 0.92
        breakdown.append("-8% (Long Commute to IT Hubs)")

    # 9. Family Friendly Rules
    if dist_school < 1: 
        price *= 1.04
        breakdown.append("+4% (Extremely Close to Schools)")
    if dist_hospital < 1: 
        price *= 1.03
        breakdown.append("+3% (Extremely Close to Hospitals)")

    # 10. Road Quality Logic
    if road_width > 40: 
        price *= 1.05
        breakdown.append("+5% (Wide Access Road)")
    elif road_width < 20: 
        price *= 0.92
        breakdown.append("-8% (Narrow Access Road)")

    # 11. Security Logic
    if gated == "Yes": 
        price *= 1.05
        breakdown.append("+5% (Gated Society Security)")
    if lift == "Yes": 
        price *= 1.03
        breakdown.append("+3% (Lift Available)")
    if floor_no > 3 and lift == "No": 
        price *= 0.90
        breakdown.append("-10% (No Lift for High Floor)")

    # 12. Legal & Trust Rules
    if rera == "Yes": 
        price *= 1.03
        breakdown.append("+3% (RERA Registered/Verified)")
    if bank_app == "Yes": 
        price *= 1.02
        breakdown.append("+2% (Bank Approved Project)")

    # 13. Furnishing Logic
    if furnishing == "Fully Furnished": 
        price *= 1.05
        breakdown.append("+5% (Fully Furnished)")
    elif furnishing == "Semi-Furnished": 
        price *= 1.02
        breakdown.append("+2% (Semi Furnished)")

    # 14. Crime Rate Logic
    if crime_rate < 20: 
        price *= 1.05
        breakdown.append("+5% (Very Safe Neighborhood)")
    elif crime_rate > 60: 
        price *= 0.90
        breakdown.append("-10% (High Crime Rate Penalty)")

    # 15. Luxury Detection Rule
    if bhk >= 4 and super_area > 2500: 
        price *= 1.12
        breakdown.append("+12% (Ultra-Luxury Layout Detected)")

    # 16. Micro Location Logic
    if dist_city < 5: 
        price *= 1.07
        breakdown.append("+7% (City Center Proximity)")
    elif dist_city > 20: 
        price *= 0.90
        breakdown.append("-10% (Far Suburb Location)")

    # 17. Property Type Logic
    if property_type == "Villa": 
        price *= 1.20
        breakdown.append("+20% (Villa Premium)")
    elif property_type == "Independent House": 
        price *= 1.15
        breakdown.append("+15% (Independent House Premium)")

    # 18. Ownership Type Logic
    if ownership_type == "Freehold": 
        price *= 1.04
        breakdown.append("+4% (Freehold Ownership)")
    elif ownership_type == "Leasehold": 
        price *= 0.94
        breakdown.append("-6% (Leasehold Restrictions)")

    # 19. Transaction Logic
    if transaction_type == "New": 
        price *= 1.03
        breakdown.append("+3% (Brand New Property)")
    elif transaction_type == "Resale": 
        price *= 0.97
        breakdown.append("-3% (Resale Depreciation)")

    # 20. Density Logic
    if total_floors > 40: 
        price *= 0.97
        breakdown.append("-3% (Overcrowded High-Rise Penalty)")
        
    # 21. Tier Strictness Override
    if locality_tier == 'Tier 2': 
        price *= 0.85
        breakdown.append("-15% (Tier 2 City Adjustment)")
    elif locality_tier == 'Tier 3': 
        price *= 0.70
        breakdown.append("-30% (Tier 3 City Adjustment)")

    return price, breakdown

st.markdown("---")
# Centered predict button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔮 Predict House Price", use_container_width=True):
        if floor_no > total_floors:
            st.error("Error: Floor Number cannot be greater than Total Floors in Building.")
        elif carpet_area > super_area:
            st.error("Error: Carpet Area cannot be greater than Super Area.")
        elif super_area < 100 or carpet_area < 50:
            st.error("Error: Area is too small to be a valid residential property.")
        elif floor_no < 0 or total_floors <= 0:
            st.error("Error: Floors must be valid continuous numbers.")
        elif bhk <= 0 or bathrooms <= 0:
            st.error("Error: A house must have at least 1 BHK and 1 Bathroom.")
        elif balcony_count > bhk + 2:
            st.warning("Warning: Unusually high number of balconies for this BHK.")
        elif property_type in ["Independent House", "Villa"] and total_floors > 5:
            st.warning("Warning: Villas and Independent Houses rarely have more than 5 floors.")
        else:
            with st.spinner("Calculating via XGBoost..."):
                base_price = predict_price()
                final_price, breakdown = apply_business_rules(base_price)
                
                st.success(f"### 🏠 Estimated Market Price: ₹ {final_price:,.2f}")
                
                # Render the Score Breakdown beautifully
                if breakdown:
                    st.markdown("#### 🤖 Real Estate AI Scoring Breakdown:")
                    st.caption("Here is exactly how the base machine learning model valuation was dynamically adjusted based on market realities:")
                    for rule in breakdown:
                        if "+" in rule:
                            st.markdown(f"**<span style='color: #4CAF50;'>{rule}</span>**", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**<span style='color: #F44336;'>{rule}</span>**", unsafe_allow_html=True)
                
                # --------- SAVE TO CSV LOGIC ---------
                # Convert the specific user choices into a dictionary for logging
                log_data = {
                    'City': city,
                    'Locality_Tier': locality_tier,
                    'BHK': bhk,
                    'Bathrooms': bathrooms,
                    'Super_Area_sqft': super_area,
                    'Carpet_Area_sqft': carpet_area,
                    'Floor_No': floor_no,
                    'Total_Floors': total_floors,
                    'Property_Age_years': property_age,
                    'Property_Type': property_type,
                    'Distance_to_IT_Hub_km': dist_it_hub,
                    'Base_ML_Price': round(base_price, 2),
                    'Predicted_Price_INR': round(final_price, 2)
                }
                
                df_log = pd.DataFrame([log_data])
                file_name = "prediction_logs.csv"
                
                import os
                if not os.path.exists(file_name):
                    df_log.to_csv(file_name, index=False)
                else:
                    df_log.to_csv(file_name, mode='a', header=False, index=False)
                
                st.info("💾 Prediction stored in `prediction_logs.csv`")


