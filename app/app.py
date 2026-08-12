import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get directory where app.py resides to construct robust relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. PAGE CONFIGURATION & TITLE
# ==========================================
st.set_page_config(page_title="CO2 Emissions Predictor", layout="centered")
st.title("🚗 Vehicle CO2 Emissions Predictor")
st.write("Input the vehicle's details below to estimate its carbon footprint.")

# ==========================================
# 2. OPTIMIZED DATA & MODEL CACHING
# ==========================================
@st.cache_resource
def load_model_artifacts():
    model_path = os.path.join(BASE_DIR, 'co2_model.pkl')
    columns_path = os.path.join(BASE_DIR, 'model_columns.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(columns_path, 'rb') as f:
        model_columns = pickle.load(f)
    return model, model_columns

@st.cache_data
def load_market_data():
    csv_path = os.path.join(BASE_DIR, '..', 'data', 'co2_emissions.csv')
    return pd.read_csv(csv_path)

try:
    model, model_columns = load_model_artifacts()
    df_display = load_market_data()
except FileNotFoundError:
    st.error("⚠️ Required system files ('co2_model.pkl', 'model_columns.pkl', or 'co2_emissions.csv') were not found!")
    st.stop()

# ==========================================
# 3. USER INTERFACE INPUT FORM
# ==========================================
st.header("📋 Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, step=1)
    fuel_consumption = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=1.0, max_value=40.0, value=8.5, step=0.1)

with col2:
    vehicle_class = st.selectbox("Vehicle Class", [
        'COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'SUV - STANDARD', 
        'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL', 'PICKUP TRUCK - STANDARD', 
        'MINIVAN', 'VAN - PASSENGER', 'SPECIAL PURPOSE VEHICLE', 'MINICOMPACT', 
        'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL', 'VAN - CARGO'
    ])
    
    transmission = st.selectbox("Transmission", ['AS', 'M', 'AV', 'A', 'AM'])
    fuel_type = st.selectbox("Fuel Type", ['Z', 'X', 'E', 'D'])

# ==========================================
# 4. FIXED INFERENCE ENGINE (Handles drop_first safely)
# ==========================================
if st.button("Predict CO2 Emissions", type="primary"):
    
    # 1. Start with an empty DataFrame matching the model's exact expected structure shape
    final_input = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 2. Map numerical features
    final_input['engine_size'] = engine_size
    final_input['cylinders'] = int(cylinders)
    final_input['fuel_consumption_comb(l/100km)'] = fuel_consumption
    
    # 3. Construct category keys
    target_vehicle_class = f"vehicle_class_{vehicle_class}"
    target_transmission = f"transmission_{transmission}"
    target_fuel_type = f"fuel_type_{fuel_type}"
    
    # 4. Safety Check: Only flip column to 1 if it exists (fixes drop_first KeyError!)
    if target_vehicle_class in final_input.columns:
        final_input[target_vehicle_class] = 1
        
    if target_transmission in final_input.columns:
        final_input[target_transmission] = 1
        
    if target_fuel_type in final_input.columns:
        final_input[target_fuel_type] = 1
        
    # 5. Make the prediction
    prediction = model.predict(final_input)[0]
    
    # Display primary prediction callout metric
    st.markdown("---")
    st.subheader("📊 Results")
    st.metric(
        label="Estimated CO2 Emissions", 
        value=f"{prediction:.2f} g/km",
        delta=f"± 3.02 g/km (Model MAE Variance)"
    )
    
    # ==========================================
    # 5. MARKET ANALYSIS DASHBOARD
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Market Insights & Comparison")
    
    tab1, tab2 = st.tabs(["Engine Size vs. Emissions", "Fuel Type Breakdown"])
    
    with tab1:
        st.write("### Where does your configuration sit on the market spectrum?")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_display, x='engine_size', y='co2_emissions', alpha=0.2, color='gray', ax=ax, label='Market Fleet')
        ax.scatter(engine_size, prediction, color='red', marker='*', s=350, zorder=5, label='Your Configuration')
        ax.set_title("Engine Displacement (L) vs CO2 Emissions")
        ax.set_xlabel("Engine Size (L)")
        ax.set_ylabel("CO2 Emissions (g/km)")
        ax.legend()
        st.pyplot(fig)
        
    with tab2:
        st.write("### Sector-wide Average Emissions by Fuel Type")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        avg_emissions = df_display.groupby('fuel_type')['co2_emissions'].mean().sort_values()
        colors = ['teal' if idx != fuel_type else 'crimson' for idx in avg_emissions.index]
        sns.barplot(x=avg_emissions.index, y=avg_emissions.values, palette=colors, ax=ax2)
        ax2.set_title("Average Structural CO2 Spread across Fuel Variants")
        ax2.set_xlabel("Fuel Classification")
        ax2.set_ylabel("Average Emissions (g/km)")
        ax2.pyplot(fig2)
