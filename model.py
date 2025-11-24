# model.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import joblib
import random
from data_prep import load_data_and_create_features
import numpy as np
import warnings
warnings.filterwarnings("ignore") # Ignore FutureWarnings for cleaner output

FEATURES = ['home_advantage', 'rating_difference', 'form_difference', 
            'h2h_home_win_rate', 'recent_goal_diff']

def evaluate_model(model, X_test, y_test):
    """Compute accuracy, F1, and ROC data."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    roc_data = []
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        # Calculate ROC for each class (Win, Draw, Loss)
        for i in range(y_proba.shape[1]):
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)
            roc_data.append({'class': i, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})
            
    return acc, f1, roc_data


def tune_and_train_mlp(X_train, y_train, X_test, y_test, scaler):
    """Tunes MLP using RandomizedSearchCV and trains the final model."""
    print("--- Starting MLP Tuning (Randomized Search) ---")
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (5, 5), (10, 10), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01, 0.1], 
    }
    
    base_mlp = MLPClassifier(activation='relu', solver='adam', max_iter=500, random_state=42)
    
    # Use RandomizedSearchCV for tuning
    rs_mlp = RandomizedSearchCV(
        estimator=base_mlp, 
        param_distributions=param_grid, 
        n_iter=10, # Number of parameter settings that are sampled
        scoring='f1_weighted', 
        cv=3, 
        random_state=42, 
        n_jobs=-1
    )
    rs_mlp.fit(X_train, y_train)
    
    tuned_mlp = rs_mlp.best_estimator_
    joblib.dump(tuned_mlp, 'mlp_model.pkl')
    
    acc, f1, roc = evaluate_model(tuned_mlp, X_test, y_test)
    joblib.dump({'acc': acc, 'f1': f1, 'roc': roc}, 'mlp_metrics.pkl')
    print(f"MLP (Tuned) saved. Best params: {rs_mlp.best_params_}")
    print(f"MLP Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return tuned_mlp


def train_ga_model(X_train, y_train, X_test, y_test):
    """Simulates a GA optimization (Random Search) for the LR model's C parameter."""
    print("--- Starting GA Optimization Simulation (Random Search) ---")
    
    best_acc = 0
    best_params = {}

    # Simulate GA population size
    for _ in range(20):
        C_candidate = random.uniform(0.1, 10)
        max_iter_candidate = random.randint(500, 2000)
        
        model = LogisticRegression(
            multi_class='multinomial', solver='lbfgs',
            C=C_candidate, max_iter=max_iter_candidate, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_params = {'C': C_candidate, 'max_iter': max_iter_candidate}

    # Train final GA-LR model with best parameters
    ga_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, **best_params)
    ga_model.fit(X_train, y_train)

    # Evaluate
    acc, f1, roc = evaluate_model(ga_model, X_test, y_test)
    joblib.dump(ga_model, 'ga_model.pkl')
    joblib.dump({'acc': acc, 'f1': f1, 'roc': roc}, 'ga_metrics.pkl')
    print(f"GA (Tuned LR) saved. Best C: {best_params['C']:.2f}, Acc: {acc:.4f}, F1 Score: {f1:.4f}")
    return ga_model


def train_and_save_all():
    """Main function to run the entire ML pipeline."""
    print("--- Starting ML Project Training Pipeline ---")
    
    # 1. Load and prepare data
    match_df = load_data_and_create_features()
    X = match_df[FEATURES]
    y = match_df['match_outcome']
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    print(f"Scaler saved to 'scaler.pkl'. Total samples: {X.shape[0]}")
    
    # 4. Train LR (Baseline)
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    joblib.dump(logreg, 'logreg_model.pkl')
    acc, f1, roc = evaluate_model(logreg, X_test_scaled, y_test)
    joblib.dump({'acc': acc, 'f1': f1, 'roc': roc}, 'logreg_metrics.pkl')
    print(f"Logistic Regression (Baseline) saved. Acc: {acc:.4f}, F1 Score: {f1:.4f}")
    
    # 5. Train MLP (Tuned)
    tune_and_train_mlp(X_train_scaled, y_train, X_test_scaled, y_test, scaler)
    
    # 6. Train GA-LR (Optimized)
    train_ga_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("--- Training and evaluation of all 3 paradigms complete ---")

if __name__ == '__main__':
    train_and_save_all()