# Explainable ECG Anomaly Detection: Final Presentation Storyline

**Time Allotment:** 8-9 minutes total
**Focus:** Live interaction and walkthrough of the central dashboard.
**Audience:** Professor and fellow students.

---

## 1. Introduction (1.5 - 2 Minutes)
* **Problem Statement:** Start with a brief context on why detecting ECG anomalies matters (e.g., catching arrhythmias early saves lives). 
* **The Challenge:** Machine learning models are often black boxes. In medical domains, doctors cannot blindly trust a prediction; they need to know *why* the model made that decision.
* **Our Solution:** An interactive dashboard that combines raw ECG data with anomaly scoring and local explainers (like Grad-CAM/SHAP overlays) seamlessly on one screen. Explain that the validation data was strictly unseen test data, ensuring methodological correctness.

## 2. Dashboard Structure Overview (1 Minute)
*(Switch directly to the Streamlit Dashboard. Do not stay on slides.)*
* **The Layout:** Point out that you designed an "all-in-one" unified view. 
* **Dataset & Performance Metrics (Top Expander):** Quickly click the `Dataset Overview & Model Performance` expander. 
    * Point out the total test series and test accuracy. 
    * Explain your choice of metric: *"We optimized for F1-Score/Recall rather than base accuracy due to the class imbalance typical in medical datasets."*
    * Close the expander to return focus to the main visualization.

## 3. Live Walkthrough: Core Scenarios (4-5 Minutes)

This is the most crucial part. Run three different test scenarios to prove your dashboard is a robust, interactive explainability tool.

### Scenario A: Normal ECG (1 Minute)
* **Action:** Select "Normal ECG" and click Generate.
* **Talking point:** "Here is a standard, healthy signal. The model outputs a low anomaly score. Notice how the explainability overlay (the heat markers) remains low or dispersed across the waveform since no specific feature stands out as anomalous."

### Scenario B: True Positive Anomaly (1.5 Minutes)
* **Action:** Select "Anomalous ECG" and click Generate.
* **Talking point:** "Here we inject a patient's signal containing an arrhythmia/spike. The model immediately catches it, flagging 'Anomalous'."
* **Interaction:** 
    * Point to the right-side visualization. 
    * Use Streamlit's/Plotly's built-in **Range Slider** below the X-axis to zoom strictly into the anomaly peak. 
    * Show how your Local Explainer (the red highlights overlaying the plot) perfectly maps to the abnormal physiological spike. *"This allows a cardiologist to bypass the noise and look exactly where the AI is pointing."*

### Scenario C: The False Positive / Error Simulation (2 Minutes)
*(Crucial component per Professor's email)*
* **Action:** Select "Simulate False Positive (Error)" and click Generate.
* **Talking point:** "Perfect performance doesn't exist in production. Here we showcase an incorrect prediction—a normal heartbeat falsely flagged as an anomaly."
* **The Explanation:** Show how the dashboard handles this gracefully. 
    * "Look at the explainer. Even though the model predicted 'Anomaly' with an artificially high score, the heatmap highlights a completely normal section, or a minor artifact/noise."
    * "This is exactly why explainability is required. A doctor reviewing this chart can see the model is fixated on noise and easily overrule the False Positive, preventing incorrect treatment. Our tool effectively debugs the model in real-time."

## 4. Conclusion & Q&A (1 Minute)
* Summarize the impact: You transformed a static neural network output into an interactive, visual decision-support tool.
* Reiterate that Shneiderman's visual analytics mantra was achieved by combining the entire ML loop—Instance, Prediction, and Explanation—into one scroll-free view.
* Open the floor for questions.