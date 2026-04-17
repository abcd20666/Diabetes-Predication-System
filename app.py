import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diabetes AI Dashboard", page_icon="🩺", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #ff4b4b;
    color: white;
    font-size: 16px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes AI Health Dashboard")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    try:
        base = os.path.dirname(__file__)
        model = joblib.load(os.path.join(base, "diabetes_model.pkl"))
    except:
        model = None

    try:
        scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    except:
        scaler = None

    return model, scaler

model, scaler = load_assets()

if model is None:
    st.error("❌ Model file not found")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Patient Input")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 110)
bp = st.sidebar.slider("Blood Pressure", 0, 130, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("DPF", 0.0, 3.0, 0.47)
age = st.sidebar.slider("Age", 21, 100, 30)

predict_btn = st.sidebar.button("🚀 Predict Now")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🔍 Prediction", "📈 Insights"])

# ---------------- DASHBOARD ----------------
with tab1:
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Glucose", glucose)
    c2.metric("BMI", bmi)
    c3.metric("Age", age)
    c4.metric("Blood Pressure", bp)

    st.markdown("---")

    # Radar Chart
    radar_fig = go.Figure()

    radar_fig.add_trace(go.Scatterpolar(
        r=[glucose, bp, bmi, age],
        theta=["Glucose", "BP", "BMI", "Age"],
        fill='toself'
    ))

    radar_fig.update_layout(title="Health Radar Chart")

    st.plotly_chart(radar_fig, use_container_width=True)

# ---------------- PREDICTION ----------------
with tab2:
    st.subheader("🔍 Prediction Result")

    if predict_btn:

        input_data = np.array([[pregnancies, glucose, bp, skin,
                                insulin, bmi, dpf, age]])

        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        try:
            prob = model.predict_proba(input_data)[0][1] * 100
        except:
            prob = 100 if prediction == 1 else 0

        col1, col2 = st.columns([2,1])

        with col1:
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={'text': "Risk %"},
                gauge={
                    'axis': {'range': [0,100]},
                    'steps': [
                        {'range':[0,30],'color':'green'},
                        {'range':[30,70],'color':'yellow'},
                        {'range':[70,100],'color':'red'}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error("🚩 High Risk")
            else:
                st.success("✅ Low Risk")

        # Recommendations
        with col2:
            st.subheader("💡 Advice")

            if glucose > 140:
                st.warning("Reduce sugar intake")
            if bmi > 30:
                st.warning("Weight control needed")
            if age > 45:
                st.warning("Regular checkups")
            if bp > 90:
                st.warning("Monitor BP")

            st.info("✔ Exercise\n✔ Healthy Diet")

    else:
        st.info("Click Predict to see results")

# ---------------- INSIGHTS ----------------
with tab3:
    st.subheader("📈 Data Insights")

    # Bar Chart
    df = {
        "Feature": ["Glucose", "BP", "BMI", "Age"],
        "Value": [glucose, bp, bmi, age]
    }

    fig1 = px.bar(df, x="Feature", y="Value", title="Health Metrics")
    st.plotly_chart(fig1, use_container_width=True)

    # Pie Chart
    risk_val = glucose + bmi
    safe_val = 200 - risk_val

    fig2 = px.pie(
        values=[risk_val, safe_val],
        names=["Risk", "Safe"],
        title="Risk Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Line Chart (dummy trend)
    trend = px.line(
        x=[1,2,3,4,5],
        y=[glucose, glucose-5, glucose+3, glucose-2, glucose],
        title="Glucose Trend (Sample)"
    )

    st.plotly_chart(trend, use_container_width=True)
