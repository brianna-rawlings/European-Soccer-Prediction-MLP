# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Soccer Match Predictor", layout="wide")

# --- Model & Metric Loading ---
@st.cache_resource
def load_assets():
    """Loads all models, metrics, and scaler."""
    try:
        models = {
            'Logistic Regression (Baseline)': joblib.load('logreg_model.pkl'),
            'MLP (Tuned)': joblib.load('mlp_model.pkl'),
            'GA Classifier': joblib.load('ga_model.pkl')
        }
        metrics = {
            'Logistic Regression (Baseline)': joblib.load('logreg_metrics.pkl'),
            'MLP (Tuned)': joblib.load('mlp_metrics.pkl'),
            'GA Classifier': joblib.load('ga_metrics.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        return models, metrics, scaler
    except FileNotFoundError:
        st.error("Assets not found. Please run 'python model.py' first to train and save all models and metrics.")
        st.stop()

models, metrics, scaler = load_assets()
FEATURES = ['home_advantage', 'rating_difference', 'form_difference', 'h2h_home_win_rate', 'recent_goal_diff']

# --- ROC Curve Plotting Function ---
def plot_roc_curves(metrics):
    """Generates a comparison plot of ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, metric_data in metrics.items():
        if metric_data['roc']:
            # Plot the average AUC (or class 2 for simplicity)
            avg_auc = np.mean([d['auc'] for d in metric_data['roc']])
            
            # Use the ROC data for Home Win (Class 2) for plotting, or the highest AUC class
            home_win_roc = [d for d in metric_data['roc'] if d['class'] == 2]
            if home_win_roc:
                ax.plot(home_win_roc[0]['fpr'], home_win_roc[0]['tpr'], 
                        label=f'{model_name} (AUC: {avg_auc:.3f})', 
                        lw=2)

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (AUC: 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Model ROC Curve Comparison (vs Home Win Class)')
    ax.legend(loc="lower right")
    st.pyplot(fig)


# --- UI Implementation ---
st.title("European Soccer Match Predictor")
st.markdown("### Comparing 3 Tuned Machine Learning Paradigms")

# --- Sidebar Inputs ---
st.sidebar.header("Match Feature Inputs (5 Core Features)")

# Feature 1: Home Advantage (Constant)
home_advantage = 1 
st.sidebar.markdown(f"**Home Advantage:** Fixed at {home_advantage}")

# Feature 2: Static Rating Difference
st.sidebar.write("### 1. Static Rating Difference")
rating_difference = st.sidebar.slider(
    'Rating Difference (Home - Away Avg Quality)',
    min_value=-20.0, max_value=20.0, value=0.0, step=0.5
)

# Feature 3: Recent Form Difference
st.sidebar.write("### 2. Recent Form Difference")
form_difference = st.sidebar.slider(
    'Form Difference (Home - Away Last 5 Matches)',
    min_value=-15.0, max_value=15.0, value=0.0, step=1.0 
)

# Feature 4: H2H Win Rate (User controls the rate based on historical data)
st.sidebar.write("### 3. H2H Win Rate")
h2h_win_rate = st.sidebar.slider(
    'H2H Win Rate (Home Team Historical Win Rate vs Opponent)',
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

# Feature 5: Recent Goal Difference
st.sidebar.write("### 4. Recent Goal Difference")
recent_goal_diff = st.sidebar.slider(
    'Recent Goal Diff (Home Team Recent Goals For - Against)',
    min_value=-2.0, max_value=2.0, value=0.0, step=0.1
)

# --- Comparison Section ---
st.header("1. Model Performance Comparison")

# Create comparison table
comparison_data = {}
for name, metric_data in metrics.items():
    avg_auc = np.mean([d['auc'] for d in metric_data['roc']]) if metric_data['roc'] else 0.0
    comparison_data[name] = {
        'Accuracy': f"{metric_data['acc']:.4f}",
        'F1 Score (Weighted)': f"{metric_data['f1']:.4f}",
        'AUC Score (Average)': f"{avg_auc:.4f}",
    }

comparison_df = pd.DataFrame.from_dict(comparison_data, orient='index')
st.dataframe(comparison_df)

st.header("2. Prediction for Hypotheical Match")

# 4. Prediction Logic
selected_model_name = st.selectbox(
    '**Select Machine Learning Paradigm to Test**', 
    list(models.keys())
)

if st.button('Predict Match Outcome'):
    model = models[selected_model_name]
    
    # Prepare the input data structure with ALL 5 features
    input_data = pd.DataFrame(
        [[home_advantage, rating_difference, form_difference, h2h_win_rate, recent_goal_diff]], 
        columns=FEATURES 
    )

    # Scale the input
    X_new_scaled = scaler.transform(input_data)
    
    # Model Prediction
    probabilities = model.predict_proba(X_new_scaled)[0]
    predicted_class = model.predict(X_new_scaled)[0]
    
    outcome_map = {2: 'HOME WIN', 1: 'DRAW', 0: 'AWAY WIN'}
    
    st.subheader(f"Prediction using: {selected_model_name}")
    final_prediction = outcome_map[predicted_class]
    st.markdown(f"**Most Likely Outcome:** <span style='color:green; font-size: 24px;'>{final_prediction}</span>", unsafe_allow_html=True)

    # Display probabilities
    prob_df = pd.DataFrame({
        'Outcome': ['AWAY WIN', 'DRAW', 'HOME WIN'],
        'Probability': probabilities
    }, index=['AWAY WIN', 'DRAW', 'HOME WIN'])
    
    st.bar_chart(prob_df['Probability'], height=300)


st.header("3. ROC Curve Analysis")
plot_roc_curves(metrics)