# --- Import Required Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    "Power Output",
    "Efficiency",
    "Hydrogen Consumption Rate",
    "Oxygen Consumption Rate",
    "Water Production",
    "Heat Generation Rate",
    "Power Density",
    "Current Density",
]

# --- Streamlit UI Design ---
st.set_page_config(page_title="Fuel Cell Load and Target Prediction", layout="wide")

# --- Sidebar: Set Constant Parameters ---
st.sidebar.markdown("<h1 style='text-align: center;'>⚡️</h1>", unsafe_allow_html=True)
st.sidebar.title("Set Constant Parameters")
FARADAY_CONSTANT = st.sidebar.number_input(
    "Faraday Constant (C/mol)", min_value=90000.0, max_value=100000.0, value=96485.0, step=0.01
)
STACK_RATED_POWER = st.sidebar.number_input(
    "Stack Rated Power (W)", min_value=50.0, max_value=200.0, value=100.0, step=0.01
)
NUMBER_OF_CELLS = st.sidebar.number_input(
    "Number of Cells", min_value=10, max_value=50, value=20, step=1
)
STACK_VOLTAGE = st.sidebar.number_input(
    "Stack Voltage (V)", min_value=6.0, max_value=24.0, value=12.0, step=0.01
)
EFFECTIVE_AREA_PEM_CELL = st.sidebar.number_input(
    "Effective Area of PEM Cell (cm²)", min_value=10.0, max_value=50.0, value=26.0, step=0.1
)
TIME_CONVERSION = st.sidebar.number_input(
    "Time Conversion (sec/min)", min_value=10.0, max_value=120.0, value=60.0, step=0.1
)
MOLE_TO_VOLUME_CONVERSION = st.sidebar.number_input(
    "Mole to Volume Conversion (L/mol)", min_value=20.0, max_value=30.0, value=23.65, step=0.01
)
REFERENCE_VOLTAGE_HEATING = st.sidebar.number_input(
    "Reference Voltage for Heating (V)", min_value=1.0, max_value=2.0, value=1.25, step=0.01
)
MAXIMUM_FUEL_CELL_VOLTAGE = st.sidebar.number_input(
    "Maximum Fuel Cell Voltage (V)", min_value=1.0, max_value=2.0, value=1.48, step=0.01
)
AVERAGE_CELL_VOLTAGE = STACK_VOLTAGE / NUMBER_OF_CELLS

# --- Main Section: Title and Input Fields ---
st.markdown(
    "<h1 style='text-align: center; color:#2196F3;'>⚡ FuelCell AI Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center;'>🎛 Input Voltage and Current</h3>",
    unsafe_allow_html=True,
)

# --- Input Section for Voltage and Current ---
col1, col2 = st.columns(2)
with col1:
    voltage_input = st.number_input(
        "Enter Voltage (V)", min_value=0.0, max_value=20.0, value=7.5, step=0.01, format="%.2f"
    )
with col2:
    current_input = st.number_input(
        "Enter Current (A)", min_value=0.0, max_value=10.0, value=6.5, step=0.01, format="%.2f"
    )

# --- Predict Load and Targets ---
if st.button("🔍 Predict "):
    input_data = np.array([[voltage_input, current_input]])
    input_scaled = scaler.transform(input_data)

    # --- Load Prediction ---
    predicted_load = rf_load.predict(input_scaled).round().astype(int)[0]

    # --- Target Predictions ---
    predicted_targets = rf_target.predict(input_scaled)[0]

    # --- Display Results ---
    st.success(f"📦 Predicted Load Condition: {predicted_load}")

    # --- Display Target Predictions ---
    st.markdown("### 📈 Predicted Values:")
    target_data = {
        "Variable": target_columns,
        "Predicted Value": [f"{predicted_targets[i]:.2f}" for i in range(len(target_columns))],
    }
    target_df = pd.DataFrame(target_data)

    # --- Centering the DataFrame ---
    st.markdown(
        target_df.style.set_table_styles(
            [{"selector": "th", "props": [("text-align", "center")]}]
        ).to_html(),
        unsafe_allow_html=True,
    )

# --- Sidebar Information ---
#st.sidebar.markdown("---")
#st.sidebar.info("")
