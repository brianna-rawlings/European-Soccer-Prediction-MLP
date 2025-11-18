# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import joblib
from data_prep import load_data_and_create_features
import random

# Features to use
FEATURES = ['home_advantage', 'rating_difference', 'form_difference']

def train_ga_model(X_train, y_train, X_test, y_test):
    """
    Simulate a GA optimization by randomly searching for the best C parameter for Logistic Regression.
    Returns the trained model and metrics.
    """
    print("--- Starting GA ---")
    
    best_acc = 0
    best_C = 1.0
    best_iter = 500

    # Random search loop simulating GA
    for _ in range(20):  # population size
        C_candidate = random.uniform(0.01, 10)
        max_iter_candidate = random.randint(200, 1000)
        
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=C_candidate,
            max_iter=max_iter_candidate,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_C = C_candidate
            best_iter = max_iter_candidate

    # Train final model with best parameters
    ga_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=best_C,
        max_iter=best_iter,
        random_state=42
    )
    ga_model.fit(X_train, y_train)

    # Evaluate
    acc, f1, roc = evaluate_model(ga_model, X_test, y_test)
    print(f"GA model trained: C={best_C:.2f}, max_iter={best_iter}, acc={acc:.4f}")
    
    return ga_model, {'acc': acc, 'f1': f1, 'roc': roc}


def train_model(model, X_train, y_train):
    """Train a given model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute accuracy, F1, and ROC data."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    roc_data = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        roc_data = []
        for i in range(y_proba.shape[1]):
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)
            roc_data.append((i, fpr, tpr, roc_auc))
    return acc, f1, roc_data

def train_and_save_all():
    """Train both Logistic Regression and MLP, save models, scaler, and evaluation metrics."""
    print("--- Starting ML Project Training Pipeline ---")
    
    # Load and prepare data
    match_df = load_data_and_create_features()
    X = match_df[FEATURES]
    y = match_df['match_outcome']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to 'scaler.pkl'")
    
    # Logistic Regression
    logreg = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    logreg = train_model(logreg, X_train_scaled, y_train)
    joblib.dump(logreg, 'logreg_model.pkl')
    print("Logistic Regression model saved.")
    
    # Evaluate LR
    acc_lr, f1_lr, roc_lr = evaluate_model(logreg, X_test_scaled, y_test)
    joblib.dump({'acc': acc_lr, 'f1': f1_lr, 'roc': roc_lr}, 'logreg_metrics.pkl')
    print(f"LR Accuracy: {acc_lr:.4f}, F1 Score: {f1_lr:.4f}")
    
    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(10,10),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    mlp = train_model(mlp, X_train_scaled, y_train)
    joblib.dump(mlp, 'mlp_model.pkl')
    print("MLP model saved.")

    # GA model
    ga_model, ga_metrics = train_ga_model(X_train_scaled, y_train, X_test_scaled, y_test)
    joblib.dump(ga_model, 'ga_model.pkl')
    joblib.dump(ga_metrics, 'ga_metrics.pkl')
    print("GA")
    print(f"GA Accuracy: {ga_metrics['acc']:.4f}, F1 Score: {ga_metrics['f1']:.4f}")

    
    # Evaluate MLP
    acc_mlp, f1_mlp, roc_mlp = evaluate_model(mlp, X_test_scaled, y_test)
    joblib.dump({'acc': acc_mlp, 'f1': f1_mlp, 'roc': roc_mlp}, 'mlp_metrics.pkl')
    print(f"MLP Accuracy: {acc_mlp:.4f}, F1 Score: {f1_mlp:.4f}")
    
    print("--- Training and evaluation complete ---")

if __name__ == '__main__':
    train_and_save_all()
