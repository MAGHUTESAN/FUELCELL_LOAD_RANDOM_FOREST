# --- Import Required Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time  # For animation timing

# --- Load Models and Scaler ---
rf_load = joblib.load("rf_load_model.pkl")
rf_target = joblib.load("rf_target_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Define Constants ---
FARADAY_CONSTANT = 96485  # C/mol
STACK_RATED_POWER = 100  # W
NUMBER_OF_CELLS = 20
STACK_VOLTAGE = 12  # V
EFFECTIVE_AREA_PEM_CELL = 26  # cm²
TIME_CONVERSION = 60  # sec/min
MOLE_TO_VOLUME_CONVERSION = 23.65  # L/mol
REFERENCE_VOLTAGE_HEATING = 1.25  # V
MAXIMUM_FUEL_CELL_VOLTAGE = 1.48  # V
AVERAGE_CELL_VOLTAGE = STACK_VOLTAGE / NUMBER_OF_CELLS

# --- Define Target Columns ---
target_columns = [
    "Power Output (W)",
    "Efficiency (%)",
    "Hydrogen Consumption Rate (mol/sec)",
    "Oxygen Consumption Rate (mol/sec)",
    "Water Production (mol/sec)",
    "Heat Generation Rate (W)",
    "Power Density (W/cm²)",
    "Current Density (A/cm²)",
]

# --- Streamlit UI Design ---
st.set_page_config(page_title="Fuel Cell Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Header ---
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF; font-family: Arial;'>⚡ FuelCell AI Predictor</h1>
    <p style='text-align: center; color: #666;'>Predict fuel cell performance with real-time insights</p>
    """, 
    unsafe_allow_html=True
)

# --- Sidebar: Interactive Parameters ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    FARADAY_CONSTANT = st.slider("Faraday Constant (C/mol)", 90000.0, 100000.0, 96485.0, 0.01)
    STACK_RATED_POWER = st.slider("Stack Rated Power (W)", 50.0, 200.0, 100.0, 0.01)
    NUMBER_OF_CELLS = st.slider("Number of Cells", 10, 50, 20, 1)
    STACK_VOLTAGE = st.slider("Stack Voltage (V)", 6.0, 24.0, 12.0, 0.01)
    EFFECTIVE_AREA_PEM_CELL = st.slider("PEM Cell Area (cm²)", 10.0, 50.0, 26.0, 0.1)
    TIME_CONVERSION = st.slider("Time Conversion (sec/min)", 10.0, 120.0, 60.0, 0.1)
    MOLE_TO_VOLUME_CONVERSION = st.slider("Mole to Volume (L/mol)", 20.0, 30.0, 23.65, 0.01)
    REFERENCE_VOLTAGE_HEATING = st.slider("Ref Voltage Heating (V)", 1.0, 2.0, 1.25, 0.01)
    MAXIMUM_FUEL_CELL_VOLTAGE = st.slider("Max Fuel Cell Voltage (V)", 1.0, 2.0, 1.48, 0.01)
    AVERAGE_CELL_VOLTAGE = STACK_VOLTAGE / NUMBER_OF_CELLS
    st.markdown(f"**Avg Cell Voltage:** {AVERAGE_CELL_VOLTAGE:.2f} V")

# --- Main Section: Input ---
st.markdown("### 🎚 Input Parameters", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    voltage_input = st.slider("Voltage (V)", 0.0, 20.0, 7.5, 0.01, format="%.2f")
with col2:
    current_input = st.slider("Current (A)", 0.0, 10.0, 6.5, 0.01, format="%.2f")

# --- Prediction with Animation ---
predict_toggle = st.checkbox("Enable Real-Time Prediction", value=True)

if predict_toggle or st.button("🔍 Predict Now"):
    # --- Prepare Input Data ---
    input_data = np.array([[voltage_input, current_input]])
    input_scaled = scaler.transform(input_data)

    # --- Predictions ---
    predicted_load = rf_load.predict(input_scaled).round().astype(int)[0]
    predicted_targets = rf_target.predict(input_scaled)[0]

    # --- Animation: Loading Spinner ---
    with st.spinner("Calculating Predictions..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Simulate processing time
            progress_bar.progress(i + 1)
        time.sleep(0.5)  # Brief pause after completion

    # --- Fade-In Effect for Results ---
    result_container = st.empty()
    result_container.markdown(
        f"<h3 style='text-align: center; color: #32CD32; opacity: 0; transition: opacity 0.5s;'>📦 Predicted Load: {predicted_load}</h3>",
        unsafe_allow_html=True
    )
    time.sleep(0.1)  # Small delay before fade-in
    result_container.markdown(
        f"<h3 style='text-align: center; color: #32CD32; opacity: 1; transition: opacity 0.5s;'>📦 Predicted Load: {predicted_load}</h3>",
        unsafe_allow_html=True
    )

    # --- Display Predicted Targets with Animation ---
    st.markdown("### 📊 Prediction Results")
    target_data = {
        "Variable": target_columns,
        "Value": [f"{val:.2f}" for val in predicted_targets]
    }
    target_df = pd.DataFrame(target_data)

    # --- Animated Table Reveal ---
    table_container = st.empty()
    table_container.markdown(
        "<div style='opacity: 0; transition: opacity 0.5s;'>Table Loading...</div>",
        unsafe_allow_html=True
    )
    time.sleep(0.3)
    table_container.dataframe(
        target_df.style.set_properties(**{
            'text-align': 'center',
            'background-color': '#f5f5f5',
            'border-radius': '5px',
            'padding': '5px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#1E90FF'), ('color', 'white'), ('text-align', 'center')]}
        ]),
        use_container_width=True
    )

    # --- Interactive Visualization with Fade-In ---
    st.markdown("### 📈 Visual Insights")
    chart_container = st.empty()
    chart_container.markdown(
        "<div style='opacity: 0; transition: opacity 0.5s;'>Chart Loading...</div>",
        unsafe_allow_html=True
    )
    time.sleep(0.3)
    fig = px.bar(
        target_df, 
        x="Variable", 
        y="Value", 
        title="Predicted Metrics",
        color="Variable",
        height=400,
        text=target_df["Value"]
    )
    fig.update_traces(textposition='auto')
    fig.update_layout(showlegend=False, bargap=0.2)
    chart_container.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #888;'>Powered by xAI | Built with Streamlit</p>
    """, 
    unsafe_allow_html=True
)