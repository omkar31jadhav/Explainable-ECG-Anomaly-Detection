import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Explainable ECG Anomaly Detection",
    page_icon="❤️",
    layout="wide"
)

# --- MOCK BACKEND FUNCTIONS ---
def load_dataset_summary():
    return {"total_series": 1000, "classes": {"Normal": 800, "Anomalous": 200}}

def get_model_results():
    return {
        "accuracy": 0.95, "f1_score": 0.92,
        "epochs": list(range(1, 21)),
        "train_loss": np.linspace(0.8, 0.1, 20) + np.random.normal(0, 0.02, 20),
        "val_loss": np.linspace(0.85, 0.15, 20) + np.random.normal(0, 0.05, 20),
    }

def predict_anomaly(series, force_incorrect=False):
    import time
    time.sleep(0.5) 
    if force_incorrect:
        # Simulate a false positive
        return {"class": "Anomalous", "score": 0.88, "ground_truth": "Normal"}
    score = np.random.uniform(0.8, 0.99) if np.random.rand() > 0.5 else np.random.uniform(0.01, 0.2)
    is_anomaly = score > 0.5
    return {"class": "Anomalous" if is_anomaly else "Normal", "score": score, "ground_truth": "Anomalous" if is_anomaly else "Normal"}

def get_explanation(series, is_anomalous=False):
    expl = np.clip(np.random.normal(0.2, 0.1, len(series)), 0, 1)
    if is_anomalous:
        expl[len(series)//2:len(series)//2+20] = np.random.uniform(0.7, 1.0, 20)
    return expl

# --- HELPER FUNCTIONS FOR VISUALIZATION ---
def generate_dummy_ecg(length=200, anomaly=False):
    x = np.linspace(0, 4 * np.pi, length)
    base = np.sin(x) + 0.1 * np.random.randn(length)
    if anomaly:
        base[length//2:length//2+10] += 2.0
    return base

def plot_ecg_with_explanation(series, explanation, title="ECG with Local Explanations"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, mode='lines', name='ECG Signal', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(
        y=series, mode='markers', name='Importance',
        marker=dict(size=8, color=explanation, colorscale='Reds', showscale=True, colorbar=dict(title="Importance"))
    ))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude", margin=dict(l=0, r=0, t=30, b=0),
                      xaxis=dict(rangeslider=dict(visible=True)))
    return fig

# --- STREAMLIT UI: CENTRAL DASHBOARD ---
st.title("❤️ Explainable ECG Anomaly Detection Dashboard")
st.markdown("A unified view for input selection, model prediction, and explainability.")

# 1. Dataset & Performance Expander (Keeps layout compact)
with st.expander("📊 Dataset Overview & Model Performance (Click to expand)", expanded=False):
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    summary = load_dataset_summary()
    results = get_model_results()
    
    with sum_col1:
        st.metric("Total Test Series", summary["total_series"])
        st.metric("Global Accuracy (Test)", f"{results['accuracy']:.2%}")
    with sum_col2:
        df_classes = pd.DataFrame(list(summary["classes"].items()), columns=['Class', 'Count'])
        fig_pie = px.pie(df_classes, values='Count', names='Class', title="Test Layout Distribution", height=200)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with sum_col3:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=results["epochs"], y=results["val_loss"], mode='lines', name='Val Loss'))
        fig_loss.update_layout(title="Validation Loss Curve", height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_loss, use_container_width=True)

# 2. Input Selection
st.subheader("1. Input Instance Selection")
input_col1, input_col2 = st.columns([1, 2])

with input_col1:
    scenario = st.radio("Select Scenario:", ["Normal ECG", "Anomalous ECG", "Simulate False Positive (Error)"])
    if st.button("Generate & Select Instance"):
        if scenario == "Normal ECG":
            st.session_state['series'] = generate_dummy_ecg(anomaly=False)
            st.session_state['is_anomaly'] = False
            st.session_state['force_err'] = False
        elif scenario == "Anomalous ECG":
            st.session_state['series'] = generate_dummy_ecg(anomaly=True)
            st.session_state['is_anomaly'] = True
            st.session_state['force_err'] = False
        else:
            # Generate a normal signal but force an incorrect prediction
            st.session_state['series'] = generate_dummy_ecg(anomaly=False)
            st.session_state['is_anomaly'] = False 
            st.session_state['force_err'] = True

if 'series' in st.session_state:
    series = st.session_state['series']
    force_err = st.session_state['force_err']
    
    # 3. Model Prediction
    st.subheader("2. Model Prediction & Explainability Overlay")
    prediction = predict_anomaly(series, force_incorrect=force_err)
    explanation = get_explanation(series, is_anomalous=prediction['class']=="Anomalous")
    
    pred_col1, pred_col2 = st.columns([1, 3])
    
    with pred_col1:
        st.markdown(f"**Ground Truth:** {prediction['ground_truth']}")
        if prediction['class'] == "Anomalous":
            if force_err:
                st.error(f"**Prediction: {prediction['class']}** ⚠️ (False Positive)")
            else:
                st.warning(f"**Prediction: {prediction['class']}**")
        else:
            st.success(f"**Prediction: {prediction['class']}**")
        st.metric("Anomaly Score", f"{prediction['score']:.2f}")
        st.markdown("_Model confidence is high. Review the explanation on the right to understand why._")
        
    with pred_col2:
        # 4. Central Visualization
        fig_expl = plot_ecg_with_explanation(series, explanation, title="Input Signal + Explainability Overlay (Zoom supported)")
        st.plotly_chart(fig_expl, use_container_width=True)

