# Project Roles and Repository Setup  
Before diving into development, the **team** should first organize the project. Create a GitHub repository with three branches: **`develop`** (for active development), **`staging`** (for tested integration), and **`main`** (final). All new features should be developed in feature-specific branches off `develop`, then merged into `staging` for testing, and finally into `main` once validated. This Git workflow (feature → develop → staging → main) ensures continuous integration and clean handoffs. All team members should agree on coding standards, commit messages, and a shared project board (e.g. GitHub Issues) to track tasks.  

**Initial Team Tasks (all members):**  
- Read the project guidelines and define clear roles.  
- Set up the repository with the branch structure (`main`, `develop`, `staging`).  
- Install development environment (Python, libraries like TensorFlow/PyTorch, Streamlit, SHAP/LIME).  
- Identify candidate time-series datasets (e.g. Numenta NAB or Yahoo S5 anomaly datasets).  
- Hold a kickoff meeting to assign responsibilities and timelines.  

Following Shneiderman’s visual analytics mantra – **“overview first, zoom and filter, then details-on-demand”** – will guide the dashboard design. Each team member should keep this principle in mind: the dashboard should present a clear overview of data/model outputs, allow interactive drilling, and show detailed explanations on demand.

## Person 1 – Data and Model Engineer  
**Role:** Collect and preprocess data, build and train the anomaly detection model.  

- **Dataset Selection & Preprocessing:** Choose one or more time-series datasets with labeled anomalies. For example, use benchmarks like Numenta’s NAB or Yahoo’s Webscope S5, which contain real and synthetic series with annotated outliers. Split time series into training/validation sets, and apply sliding-window segmentation (converting series into fixed-length input windows) as discussed in anomaly detection pipelines. Clean and normalize the data (e.g. scaling, denoising).  

- **Model Development:** Implement a time-series anomaly detection model. A strong choice is a 1D Convolutional Neural Network (CNN) or a sequence model like LSTM/Autoencoder. For example, a CNN can learn temporal filters to predict normal behavior. Alternatively, an LSTM-Autoencoder can reconstruct normal sequences and flag high reconstruction error as anomalies. Ensure the model outputs an anomaly score per time point (prediction error or probability).  

- **Training and Baselines:** Train the model on “normal” (non-anomalous) data if unsupervised, or train a classifier if anomalies are labeled. Record metrics such as precision, recall, F1 and AUC-ROC on validation data. Note that anomaly detection often needs a decision threshold – using **AUC-ROC/AUC-PR** is helpful since it is threshold-independent. Person 1 should generate confusion matrices and loss curves to evaluate performance.  

- **Pipeline Integration:** Follow the generic pipeline for time-series anomaly detection: (1) Data preprocessing (windows, features), (2) Anomaly detection model, (3) Scoring each window/point, (4) Post-processing thresholding. This staged approach helps compare different methods systematically.  

- **Output:** Save the trained model (and any preprocessing code). Provide a function that takes new time series input and returns a prediction score/class for each window. Document model architecture and hyperparameters. Ensure code is modular so Person 2 (below) can use the model for explanations.  

## Person 2 – Explainability (XAI) Specialist  
**Role:** Research and implement XAI methods to explain model predictions on time series.  

- **XAI Method Research:** Investigate suitable explainability techniques for time-series models. Many popular XAI libraries (e.g. OmniXAI) support methods like **SHAP**, **LIME**, and **Grad-CAM** for time-series data. These methods generate local explanations (feature importances over time). Plan to implement at least two different explainers. For a CNN-based model, Grad-CAM can highlight important time steps; LIME/SHAP can attribute importance to input features (time points or window segments).  

- **Implementation:** Integrate XAI tools with the trained model. For each test sequence, compute explanations showing which parts of the series were most “anomalous” to the model. For example, use SHAP to get a time-series importance map or use Grad-CAM on intermediate CNN layers. Person 2 will write functions that take a raw time-series input and model prediction, and output an explanation (e.g. a heatmap over time).  

- **Validation:** Generate examples of explanations on known anomalies. Verify that the highlighted regions make sense (e.g. SHAP shows peaks at true anomalies). Document how each XAI method is applied. Since target users include non-technical domain experts, also prepare simple textual or visual summaries of what the explanation means. (For instance, “The model flagged the sudden spike here as abnormal.”)  

- **Output:** Deliver scripts/notebooks that produce explanation plots (e.g. importance vs. time) and any summary statistics (e.g. average explanation weight on anomalies vs. normals). Collaborate with Person 1 to ensure the model interface is compatible.  

## Person 3 – Dashboard and Visualization Engineer  
**Role:** Design and implement the interactive Streamlit dashboard according to visual analytics principles.  

- **UI/UX Design:** Plan the dashboard layout following **Shneiderman’s mantra**. For example:  
  - **Overview Page:** Show dataset summaries (total series count, class distribution, typical time-series plots). Use charts (histograms, line plots, pie charts) so users see the big picture.  
  - **Model Results Page:** Visualize model performance (training vs validation loss curves, ROC curve, confusion matrix) to give an overview of accuracy.  
  - **Live Prediction Page:** Provide an interface where a user can upload or select a time-series input. Display the raw series, the model’s predicted anomaly score/class, and confidence. For instance: “Prediction: Anomalous (score=0.92)”.  
  - **Explanation Page:** Show local explanations for a selected signal. Overlay highlights on the series (e.g. heatmaps from Grad-CAM/SHAP) so users see which timestamps triggered the anomaly.  
  - **Comparison Page:** Allow side-by-side comparison of a normal vs. anomalous series with explanations. This reinforces understanding (e.g. “You can see how the anomaly peak is highlighted here”).  

- **Implementation:** Use **Streamlit** (or Dash) to build the app, as recommended for quick prototyping. Add interactive widgets (file uploader, dropdowns for series selection, threshold sliders). Use Plotly or Matplotlib to draw time-series and heatmap overlays. Implement filtering/zooming on plots (enable the user to zoom into interesting intervals). For example, a Plotly line chart with a “range slider” fits the mantra of zoom/filter.  

- **Integration:** Person 3 will call Person 1’s model and Person 2’s explanation functions under the hood. Ensure the app can request a prediction and explanation for any selected input. Structure code so front-end widgets trigger back-end computation smoothly.  

- **Output:** A working Streamlit app (`app.py`) with multiple pages or tabs. Include instructions/help text for each page. Follow Shneiderman’s principle by starting each page with an overview chart, allowing drill-down. (E.g. the first visualization on the Explanation page might be the entire ECG waveform with anomalies highlighted; then a zoomed inset could show details.) No coding references needed here, but keep the design clear and visual.  

## Collaboration Workflow  
- **Parallel Development:** Each person works primarily in a separate domain (data/model, XAI, UI). They should use separate feature branches off `develop` (e.g. `feature-data`, `feature-xai`, `feature-ui`). At agreed checkpoints, merge changes into the `staging` branch and resolve conflicts.  

- **Integration Steps:** Regularly sync up. For example, once Person 1 has a working model, they push code to `develop` or a feature branch. Person 2 can then merge from `develop` to apply XAI. Person 3 similarly waits for the model & XAI code to exist in `staging` before final integration. Use pull requests to review code before merging to `staging`. This ensures that by project end, all components (model, explainers, dashboard) work seamlessly together.  

- **Documentation & Testing:** Each member should document their code and create a README or wiki sections explaining setup (datasets, model architecture, how to run explainers, how to launch the dashboard). Include example runs and screenshots in the repo. Also write a short evaluation script or Jupyter notebook that demonstrates the full pipeline: loading data, training model, running explainers, and showing dashboard screenshots or data.  

By following this plan, the team can develop in parallel: Person 1 builds and validates the core anomaly model (using a standard time-series pipeline); Person 2 implements XAI methods (leveraging tools that support SHAP/LIME/Grad-CAM for time series); and Person 3 designs the interactive dashboard guided by Shneiderman’s mantra. Along the way, cite results and visualizations with clear explanations so that even non-technical users (students, lecturers, domain experts) can understand how anomalies are detected and why each decision was made. This division of labor ensures all parts (data, model, explainability, visualization) progress together and integrate smoothly. 

**Sources:** Established practices in time-series anomaly detection and explainability inform this plan. These references cover the importance of anomaly detection, the generic detection pipeline, available XAI tools for time series, interactive dashboard principles, and evaluation metrics for anomaly detection.