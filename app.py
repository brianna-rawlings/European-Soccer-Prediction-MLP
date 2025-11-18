# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Soccer Match Predictor", layout="wide")
st.title("⚽ European Soccer Match Predictor")
st.write("---")

# --- Load Models & Metrics ---
@st.cache_resource
def load_models():
    logreg_model = joblib.load('logreg_model.pkl')
    mlp_model = joblib.load('mlp_model.pkl')
    ga_model = joblib.load('ga_model.pkl')          
    scaler = joblib.load('scaler.pkl')

    logreg_metrics = joblib.load('logreg_metrics.pkl')
    mlp_metrics = joblib.load('mlp_metrics.pkl')
    ga_metrics = joblib.load('ga_metrics.pkl')      

    return logreg_model, mlp_model, ga_model, scaler, logreg_metrics, mlp_metrics, ga_metrics

#  Load everything in one step
models_and_metrics = load_models()
logreg_model, mlp_model, ga_model, scaler, logreg_metrics, mlp_metrics, ga_metrics = models_and_metrics

# --- Model Selection ---
selected_model_name = st.selectbox(
    'Select Machine Learning Paradigm:',
    ['Logistic Regression', 'Multilayer Perceptron (MLP)', 'GA']
)

if selected_model_name == 'Logistic Regression':
    model = models_and_metrics[0]   # logreg_model
    metrics = models_and_metrics[4] # logreg_metrics
elif selected_model_name == 'Multilayer Perceptron (MLP)':
    model = models_and_metrics[1]   # mlp_model
    metrics = models_and_metrics[5] # mlp_metrics
else:
    model = models_and_metrics[2]   # ga_model
    metrics = models_and_metrics[6] # ga_metrics

st.info(f"Using {selected_model_name}")

# --- Sidebar Inputs ---
st.sidebar.header("Match Feature Inputs")
home_advantage = 1
rating_difference = st.sidebar.slider(
    "Rating Difference (Home - Away)", -20.0, 20.0, 0.0, 0.5, key="rating_diff"
)
form_difference = st.sidebar.slider(
    "Recent Form Difference (Home - Away)", -15.0, 15.0, 0.0, 1.0, key="form_diff"
)

# --- Prediction ---
if st.button("Predict Match Outcome"):
    st.subheader("Prediction Results")

    # Prepare input
    new_match_data = pd.DataFrame([[home_advantage, rating_difference, form_difference]],
                                  columns=['home_advantage', 'rating_difference', 'form_difference'])
    X_new_scaled = scaler.transform(new_match_data)

    # Predict
    probabilities = model.predict_proba(X_new_scaled)[0]
    predicted_class = model.predict(X_new_scaled)[0]
    outcome_map = {2: 'HOME WIN', 1: 'DRAW', 0: 'AWAY WIN'}
    st.markdown(f"**Predicted Outcome:** <span style='color:green; font-size: 24px;'>{outcome_map[predicted_class]}</span>", unsafe_allow_html=True)

    # Probability Bar Chart
    prob_df = pd.DataFrame({'Outcome': ['AWAY WIN', 'DRAW', 'HOME WIN'], 'Probability': probabilities}).set_index('Outcome')
    st.bar_chart(prob_df)

    st.markdown("---")

    # Metrics Display in Columns
    col1, col2 = st.columns(2)
    col1.metric(label="Test Accuracy", value=f"{metrics['acc']:.4f}")
    col2.metric(label="Test F1 Score (Weighted)", value=f"{metrics['f1']:.4f}")

    # Interactive ROC Curve with Plotly
    roc_data = metrics.get('roc', [])
    if roc_data:
        fig = go.Figure()
        class_names = ['AWAY WIN', 'DRAW', 'HOME WIN']
        for cls, fpr, tpr, roc_auc in roc_data:
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{class_names[cls]} (AUC={roc_auc:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
        fig.update_layout(
            title=f"ROC Curve — {selected_model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=700, height=500,
            legend=dict(x=0.7, y=0.2)
        )
        st.plotly_chart(fig)
    else:
        st.warning(f"{selected_model_name} does not support probability predictions for ROC.")
