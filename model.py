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
import numpy as np


# Features to use
FEATURES = [
    'home_advantage',
    'rating_difference',
    'form_difference',
    'h2h_home_win_rate',
    'recent_goal_diff'
]

# --- Helper Functions ---
def train_model(model, X_train, y_train):
    """Train a given model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute accuracy, F1, and ROC data (as dictionaries)."""
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
            roc_data.append({
                'class': i,
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            })
    return acc, f1, roc_data

# --- Genetic Algorithm for Logistic Regression ---
def train_ga_model(X_train, y_train, X_test, y_test,
                   population_size=20, generations=10, mutation_rate=0.2):
    """Genetic Algorithm to optimize Logistic Regression hyperparameters (C, max_iter)."""
    print("--- Starting True GA ---")
    
    # Initialize population: [C, max_iter]
    population = [[10 ** random.uniform(-3, 3), random.randint(200, 2000)]
                  for _ in range(population_size)]

    def fitness(individual):
        C, max_iter = individual
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=C,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    for gen in range(generations):
        fitness_scores = [fitness(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        print(f"Gen {gen+1} best acc: {fitness_scores[best_idx]:.4f}, "
              f"C={population[best_idx][0]:.3f}, max_iter={population[best_idx][1]}")
        
        # Selection + Crossover + Mutation
        new_population = []
        while len(new_population) < population_size:
            i1, i2 = random.sample(range(population_size), 2)
            parent = population[i1] if fitness_scores[i1] > fitness_scores[i2] else population[i2]

            i3 = random.randint(0, population_size-1)
            other_parent = population[i3]
            child = [
                (parent[0] + other_parent[0]) / 2,
                int((parent[1] + other_parent[1]) / 2)
            ]

            if random.random() < mutation_rate:
                child[0] *= 10 ** random.uniform(-0.2, 0.2)
                child[1] += random.randint(-100, 100)
                child[1] = max(100, min(child[1], 3000))
            
            new_population.append(child)
        population = new_population

    # Train final GA model with best individual
    fitness_scores = [fitness(ind) for ind in population]
    best_idx = np.argmax(fitness_scores)
    best_C, best_iter = population[best_idx]
    ga_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=best_C,
        max_iter=best_iter,
        random_state=42
    )
    ga_model.fit(X_train, y_train)
    acc, f1, roc = evaluate_model(ga_model, X_test, y_test)
    print(f"GA final model: C={best_C:.3f}, max_iter={best_iter}, acc={acc:.4f}")
    return ga_model, {'acc': acc, 'f1': f1, 'roc': roc}

# --- Full Training Pipeline ---
def train_and_save_all():
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
    acc_mlp, f1_mlp, roc_mlp = evaluate_model(mlp, X_test_scaled, y_test)
    joblib.dump({'acc': acc_mlp, 'f1': f1_mlp, 'roc': roc_mlp}, 'mlp_metrics.pkl')
    print(f"MLP Accuracy: {acc_mlp:.4f}, F1 Score: {f1_mlp:.4f}")

    # GA
    ga_model, ga_metrics = train_ga_model(X_train_scaled, y_train, X_test_scaled, y_test)
    joblib.dump(ga_model, 'ga_model.pkl')
    joblib.dump(ga_metrics, 'ga_metrics.pkl')
    print(f"GA Accuracy: {ga_metrics['acc']:.4f}, F1 Score: {ga_metrics['f1']:.4f}")

    print("--- Training and evaluation complete ---")

if __name__ == '__main__':
    train_and_save_all()
