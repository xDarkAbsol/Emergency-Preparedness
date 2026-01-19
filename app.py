import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("emergency_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Community Emergency Readiness Predictor")
st.write("Enter key details to predict emergency preparedness")
expected_features = ['Health_Literacy_Score', 'Emergency_Response_Satisfaction', 'Month', 'Hour',
                     'Age_Group_26–35', 'Age_Group_36–45', 'Age_Group_46–60', 'Age_Group_60+',
                     'Education_Level_Higher secondary', 'Education_Level_No formal education', 'Education_Level_Postgraduate+', 'Education_Level_School level',
                     'Occupation_Homemaker', 'Occupation_Retired/Unemployed', 'Occupation_Self-employed', 'Occupation_Student',
                     'Area_Type_Semi-urban', 'Area_Type_Slum/High-density', 'Area_Type_Urban',
                     'Understands_Doctor_Often', 'Understands_Doctor_Rarely', 'Understands_Doctor_Sometimes',
                     'Knows_Emergency_Numbers_Yes',
                     'Health_Info_Source_Family/Friends', 'Health_Info_Source_Internet', 'Health_Info_Source_Social Media', 'Health_Info_Source_TV/Radio',
                     'Aware_of_PreAlert_Yes', 'Received_Alert_Before_Yes',
                     'Preferred_Alert_Method_Phone Call', 'Preferred_Alert_Method_SMS', 'Preferred_Alert_Method_Siren/Loudspeaker',
                     'Faced_Emergency_Last5Yrs_Yes',
                     'Emergency_Response_Time_30–60 min', 'Emergency_Response_Time_<15 min', 'Emergency_Response_Time_>1 hour',
                     'Nearby_Healthcare_Available_Yes',
                     'Main_Barrier_Cost', 'Main_Barrier_Distance', 'Main_Barrier_Lack of awareness', 'Main_Barrier_Traffic',
                     'Distance_to_Facility_3–5 km', 'Distance_to_Facility_<1 km', 'Distance_to_Facility_>5 km']

input_dict = {feature: 0 for feature in expected_features}

input_dict['Month'] = 9  
input_dict['Hour'] = 11  

# Inputs
health_literacy = st.slider("1. Self-rated understanding of basic health information (1-5)", 1, 5, 3)
emergency_satisfaction = st.slider("2. Emergency Response Satisfaction (1-5)", 1, 5, 3)
pre_alert = st.selectbox("3. Aware of Pre-alert Systems", ["No", "Yes"])
distance = st.selectbox("4. Distance to Nearest Healthcare Facility", ["1–3 km", "<1 km", "3–5 km", ">5 km"])

occupation = st.selectbox(
    "5. Occupation",
    ["Employed", "Homemaker", "Retired/Unemployed", "Self-employed", "Student"]
)
main_barrier = st.selectbox(
    "6. Main Barrier to Emergency Care",
    ["No barrier", "Cost", "Distance", "Lack of awareness", "Traffic"]
)
preferred_alert_method = st.selectbox(
    "7. Preferred Alert Method",
    ["Email", "Phone Call", "SMS", "Siren/Loudspeaker", "Other"]
)
emergency_response_time = st.selectbox(
    "8. Typical Emergency Response Time",
    ["15–30 min", "<15 min", "30–60 min", ">1 hour"]
)

# Update input_dict with user's selections
input_dict['Health_Literacy_Score'] = health_literacy
input_dict['Emergency_Response_Satisfaction'] = emergency_satisfaction
input_dict['Aware_of_PreAlert_Yes'] = 1 if pre_alert == "Yes" else 0

# Distance_to_Facility: 
if distance == "<1 km":
    input_dict['Distance_to_Facility_<1 km'] = 1
elif distance == "3–5 km":
    input_dict['Distance_to_Facility_3–5 km'] = 1
elif distance == ">5 km":
    input_dict['Distance_to_Facility_>5 km'] = 1

# Occupation: 
if occupation == "Homemaker":
    input_dict['Occupation_Homemaker'] = 1
elif occupation == "Retired/Unemployed":
    input_dict['Occupation_Retired/Unemployed'] = 1
elif occupation == "Self-employed":
    input_dict['Occupation_Self-employed'] = 1
elif occupation == "Student":
    input_dict['Occupation_Student'] = 1

# Main_Barrier: 
if main_barrier == "Cost":
    input_dict['Main_Barrier_Cost'] = 1
elif main_barrier == "Distance":
    input_dict['Main_Barrier_Distance'] = 1
elif main_barrier == "Lack of awareness":
    input_dict['Main_Barrier_Lack of awareness'] = 1
elif main_barrier == "Traffic":
    input_dict['Main_Barrier_Traffic'] = 1

# Preferred_Alert_Method: 
if preferred_alert_method == "Phone Call":
    input_dict['Preferred_Alert_Method_Phone Call'] = 1
elif preferred_alert_method == "SMS":
    input_dict['Preferred_Alert_Method_SMS'] = 1
elif preferred_alert_method == "Siren/Loudspeaker":
    input_dict['Preferred_Alert_Method_Siren/Loudspeaker'] = 1

# Emergency_Response_Time: 
if emergency_response_time == "<15 min":
    input_dict['Emergency_Response_Time_<15 min'] = 1
elif emergency_response_time == "30–60 min":
    input_dict['Emergency_Response_Time_30–60 min'] = 1
elif emergency_response_time == ">1 hour":
    input_dict['Emergency_Response_Time_>1 hour'] = 1


input_df = pd.DataFrame([input_dict], columns=expected_features)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success("High Emergency Readiness" if prediction == 1 else "Low Emergency Readiness")
