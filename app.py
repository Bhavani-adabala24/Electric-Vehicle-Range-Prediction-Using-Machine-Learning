import streamlit as st
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <style>
    .stApp {
        background: #07111f;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 6px;
        margin-bottom: 30px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 700;
        margin: 8px 0 14px 0;
    }

    .prediction-card {
        min-height: 220px;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        background: linear-gradient(135deg, #12372f, #0b2425);
        border: 1px solid #22c55e;
        box-shadow: 0 12px 35px rgba(34, 197, 94, 0.12);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .prediction-title {
        color: #86efac;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .prediction-value {
        color: #ffffff;
        font-size: 52px;
        font-weight: 800;
        margin: 12px 0;
    }

    .prediction-text {
        color: #94a3b8;
        font-size: 14px;
    }

    .info-card {
        background: #0f1b2d;
        border: 1px solid #243247;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        min-height: 85px;
    }

    .info-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .info-value {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 700;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 11px;
        border: none;
        background: #22c55e;
        color: white;
        font-size: 16px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #16a34a;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: #0f1b2d;
        border: 1px solid #243247;
        border-radius: 14px;
        padding: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 35px;
    }
    </style>
    """),
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">⚡ EV Range Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Predict the estimated driving range of an Electric Vehicle using Machine Learning.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
try:
    data = pd.read_csv("ev_dataset.csv")
except FileNotFoundError:
    st.error("❌ ev_dataset.csv was not found. Keep it in the same folder as app.py.")
    st.stop()

required_columns = [
    "battery_capacity_kWh",
    "vehicle_weight_kg",
    "avg_speed_kmph",
    "temperature_C",
    "range_km",
]

missing = [column for column in required_columns if column not in data.columns]
if missing:
    st.error(f"❌ Missing columns in ev_dataset.csv: {', '.join(missing)}")
    st.stop()

# --------------------------------------------------
# Prepare and Train Model
# --------------------------------------------------
X = data[
    [
        "battery_capacity_kWh",
        "vehicle_weight_kg",
        "avg_speed_kmph",
        "temperature_C",
    ]
]
y = data["range_km"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Dashboard
# --------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-title">🚗 Vehicle Parameters</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        battery = st.number_input(
            "🔋 Battery Capacity (kWh)",
            min_value=1.0,
            max_value=250.0,
            value=80.0,
            step=1.0,
        )

        weight = st.number_input(
            "⚖️ Vehicle Weight (kg)",
            min_value=100.0,
            max_value=5000.0,
            value=1900.0,
            step=50.0,
        )

        speed = st.number_input(
            "🚗 Average Speed (km/h)",
            min_value=1.0,
            max_value=250.0,
            value=75.0,
            step=1.0,
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-30.0,
            max_value=60.0,
            value=30.0,
            step=1.0,
        )

        predict_button = st.form_submit_button("⚡ Predict EV Range")

with right:
    st.markdown(
        '<div class="section-title">📊 Prediction</div>',
        unsafe_allow_html=True,
    )

    if predict_button:
        new_ev = pd.DataFrame(
            [[battery, weight, speed, temperature]],
            columns=X.columns,
        )

        predicted_range = model.predict(new_ev)[0]

        prediction_html = textwrap.dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">Estimated Driving Range</div>
                <div class="prediction-value">{predicted_range:.2f} km</div>
                <div class="prediction-text">Based on the vehicle parameters provided</div>
            </div>
            """
        )

        st.markdown(prediction_html, unsafe_allow_html=True)

    else:
        prediction_html = textwrap.dedent(
            """
            <div class="prediction-card">
                <div class="prediction-title">Ready to Predict</div>
                <div class="prediction-value">---</div>
                <div class="prediction-text">Enter vehicle details and click Predict</div>
            </div>
            """
        )

        st.markdown(prediction_html, unsafe_allow_html=True)

# --------------------------------------------------
# Vehicle Profile
# --------------------------------------------------
st.markdown(
    '<div class="section-title" style="margin-top:30px;">📋 Vehicle Profile</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

profiles = [
    ("Battery", f"{battery:.0f} kWh"),
    ("Weight", f"{weight:.0f} kg"),
    ("Average Speed", f"{speed:.0f} km/h"),
    ("Temperature", f"{temperature:.0f} °C"),
]

for column, (label, value) in zip([c1, c2, c3, c4], profiles):
    with column:
        card = textwrap.dedent(
            f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
            </div>
            """
        )
        st.markdown(card, unsafe_allow_html=True)
# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    '<div class="footer">EV Range Prediction • Linear Regression • Python • Scikit-learn • Streamlit</div>',
    unsafe_allow_html=True,
)