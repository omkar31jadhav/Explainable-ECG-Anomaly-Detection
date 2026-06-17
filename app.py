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

# --- MOCK BACKEND FUNCTIONS (To be integrated with Person 1's and Person 2's code) ---
def load_dataset_summary():
    """Mock dataset summary."""
    return {
        "total_series": 1000,
        "classes": {"Normal": 800, "Anomalous": 200}
    }

def get_model_results():
    """Mock model evaluation metrics."""
    return {
        "accuracy": 0.95,
        "f1_score": 0.92,
        "epochs": list(range(1, 21)),
        "train_loss": np.linspace(0.8, 0.1, 20) + np.random.normal(0, 0.02, 20),
        "val_loss": np.linspace(0.85, 0.15, 20) + np.random.normal(0, 0.05, 20),
    }

def predict_anomaly(series):
    """Mock prediction function."""
    import time
    time.sleep(1) # simulate inference time
    score = np.random.uniform(0.7, 0.99) if np.random.rand() > 0.5 else np.random.uniform(0.01, 0.3)
    is_anomaly = score > 0.5
    return {"class": "Anomalous" if is_anomaly else "Normal", "score": score}

def get_explanation(series):
    """Mock explanation function (e.g., Grad-CAM / SHAP)."""
    # Simply generate random importance scores for the length of the series
    return np.clip(np.random.normal(0.5, 0.2, len(series)), 0, 1)

# --- HELPER FUNCTIONS FOR VISUALIZATION ---
def generate_dummy_ecg(length=200, anomaly=False):
    """Generates a dummy ECG-like signal for demonstration."""
    x = np.linspace(0, 4 * np.pi, length)
    base = np.sin(x) + 0.1 * np.random.randn(length)
    if anomaly:
        # Add a spike
        base[length//2:length//2+10] += 2.0
    return base

def plot_ecg_series(series, title="ECG Signal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, mode='lines', name='Signal'))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude",
                      xaxis=dict(rangeslider=dict(visible=True)))
    return fig

def plot_ecg_with_explanation(series, explanation, title="ECG with Highlighted Local Explanations"):
    fig = go.Figure()
    
    # Base signal
    fig.add_trace(go.Scatter(y=series, mode='lines', name='ECG Signal', line=dict(color='blue')))
    
    # Colored markers based on explanation score (heatmap overlay logic)
    fig.add_trace(go.Scatter(
        y=series,
        mode='markers',
        name='Importance',
        marker=dict(
            size=8,
            color=explanation,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Importance Score")
        )
    ))
    
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude",
                      xaxis=dict(rangeslider=dict(visible=True)))
    return fig

# --- STREAMLIT UI ---
st.title("❤️ Explainable ECG Anomaly Detection")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model Results", "Live Prediction", "Explanation", "Comparison"]
)

# 1. Overview Page
if page == "Overview":
    st.header("Dataset Overview")
    st.markdown("Visualizing the big picture: dataset summaries and class distributions.")
    
    summary = load_dataset_summary()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Series Count", summary["total_series"])
        # Class distribution pie chart
        df_classes = pd.DataFrame(list(summary["classes"].items()), columns=['Class', 'Count'])
        fig_pie = px.pie(df_classes, values='Count', names='Class', title="Class Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Typical Time-Series Plot")
        sample_normal = generate_dummy_ecg()
        fig_normal = plot_ecg_series(sample_normal, "Typical Normal ECG")
        st.plotly_chart(fig_normal, use_container_width=True)

# 2. Model Results Page
elif page == "Model Results":
    st.header("Model Performance")
    st.markdown("Training vs validation loss curves and overall accuracy overview.")
    
    results = get_model_results()
    
    col1, col2 = st.columns(2)
    col1.metric("Validation Accuracy", f"{results['accuracy']:.2%}")
    col2.metric("Validation F1 Score", f"{results['f1_score']:.2f}")
    
    st.subheader("Loss Curve")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=results["epochs"], y=results["train_loss"], mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=results["epochs"], y=results["val_loss"], mode='lines', name='Validation Loss'))
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)
    
    st.info("Additional charts like ROC curves and Confusion Matrices can be integrated here after final model training.")

# 3. Live Prediction Page
elif page == "Live Prediction":
    st.header("Live Prediction")
    st.markdown("Upload or select a time-series input to see the model's prediction.")
    
    input_type = st.radio("Select Input Source", ["Generate Sample", "Upload CSV"])
    series = None
    
    if input_type == "Generate Sample":
        if st.button("Generate Random ECG"):
            # Randomly pick normal or anomaly
            is_anomaly = np.random.rand() > 0.5
            series = generate_dummy_ecg(anomaly=is_anomaly)
            st.session_state['live_series'] = series
            st.success("Sample generated!")
    else:
        uploaded_file = st.file_uploader("Upload an ECG CSV file (1 column of values)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            series = df.iloc[:, 0].values
            st.session_state['live_series'] = series
            st.success("File uploaded successfully!")
            
    if 'live_series' in st.session_state:
        series = st.session_state['live_series']
        st.plotly_chart(plot_ecg_series(series, title="Raw ECG Input"), use_container_width=True)
        
        if st.button("Run Prediction"):
            with st.spinner("Analyzing..."):
                prediction = predict_anomaly(series)
            
            if prediction["class"] == "Anomalous":
                st.error(f"**Prediction: {prediction['class']}** (score={prediction['score']:.2f})")
            else:
                st.success(f"**Prediction: {prediction['class']}** (score={prediction['score']:.2f})")

# 4. Explanation Page
elif page == "Explanation":
    st.header("Local Explanations")
    st.markdown("Visualizing which timestamps triggered the anomaly through Grad-CAM or SHAP overlay heatmaps.")
    
    if st.button("Load Anomalous Example"):
        series = generate_dummy_ecg(anomaly=True)
        st.session_state['expl_series'] = series
        st.session_state['expl_scores'] = get_explanation(series)
        
    if 'expl_series' in st.session_state:
        st.subheader("Explanation Overlay (Zoom/Filter supported)")
        fig_expl = plot_ecg_with_explanation(st.session_state['expl_series'], st.session_state['expl_scores'])
        st.plotly_chart(fig_expl, use_container_width=True)
        st.info("Hover over the points to see the local importance score. Use the range slider below the x-axis to zoom into the anomaly peak.")

# 5. Comparison Page
elif page == "Comparison":
    st.header("Comparison: Normal vs. Anomalous")
    st.markdown("Side-by-side comparison to reinforce understanding of normal vs anomalous patterns.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Normal Series")
        normal_series = generate_dummy_ecg(anomaly=False)
        normal_expl = get_explanation(normal_series)
        st.plotly_chart(plot_ecg_with_explanation(normal_series, normal_expl, title="Normal"), use_container_width=True)
        
    with col2:
        st.subheader("Anomalous Series")
        anomaly_series = generate_dummy_ecg(anomaly=True)
        anomaly_expl = get_explanation(anomaly_series) # Suppose an anomaly is around the middle
        # Make the explanation spike where the anomaly is
        anomaly_expl[80:120] = np.random.uniform(0.8, 1.0, 40)
        
        st.plotly_chart(plot_ecg_with_explanation(anomaly_series, anomaly_expl, title="Anomalous (Highlighted)"), use_container_width=True)
        st.markdown("*Notice how the anomaly peak is highlighted here by the high importance scores.*")
