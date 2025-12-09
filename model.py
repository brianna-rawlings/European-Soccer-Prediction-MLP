# model.py

import pandas as pd
import numpy as np
import random
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score

# Assuming data_prep is in the same directory and handles loading/feature creation
from data_prep import load_data_and_create_features
import numpy as np
import warnings
warnings.filterwarnings("ignore") # Ignore FutureWarnings for cleaner output

# --- GLOBAL CONFIGURATION ---
RANDOM_STATE = 42 # Defined to fix the original NameError and ensure reproducibility

# Features to use (The 5 Core Features)
FEATURES = ['home_advantage', 'rating_difference', 'form_difference', 
            'h2h_home_win_rate', 'recent_goal_diff']

# --- UTILITY FUNCTION ---

def evaluate_model(model, X_test, y_test):
    """
    Returns accuracy, F1 score, and ROC curve data (if supported).
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    roc_data = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        roc_data = []

        for cls in range(y_proba.shape[1]):
            # Use 'ovr' (one-vs-rest) approach for multi-class ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, cls], pos_label=cls)
            roc_data.append({
                "class": cls,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc(fpr, tpr)
            })

    return acc, f1, roc_data


# --- GENETIC ALGORITHM OPTIMIZATION ---

def train_ga_model(
    X_train, y_train, X_val, y_val,
    population_size=30,
    generations=30,
    mutation_rate=0.2,
    elite_size=2,
    tournament_k=3,
    early_stop=5
):
    """
    Improved GA to optimize Logistic Regression hyperparameters (C, max_iter)
    using elitism, tournament selection, uniform crossover, and adaptive mutation.
    """
    print("\n--- Starting Improved Genetic Algorithm Optimization ---")

    # Initialize population: [C, max_iter]
    population = [
        [10 ** random.uniform(-3, 3), random.randint(200, 2000)]
        for _ in range(population_size)
    ]

    best_overall_score = 0
    best_overall_individual = None
    no_improve_count = 0

    def fitness(individual):
        """Fitness function using 3-fold cross-validation on training set."""
        C, max_iter = individual
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            random_state=RANDOM_STATE
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        return scores.mean()

    for gen in range(generations):
        fitness_scores = [fitness(ind) for ind in population]

        # Track best individual
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_score = fitness_scores[gen_best_idx]
        if gen_best_score > best_overall_score:
            best_overall_score = gen_best_score
            best_overall_individual = population[gen_best_idx]
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"Gen {gen+1:02d} | Best Accuracy = {gen_best_score:.4f} | "
              f"C={population[gen_best_idx][0]:.4f}, max_iter={population[gen_best_idx][1]}")

        if no_improve_count >= early_stop:
            print(f"No improvement for {early_stop} generations. Early stopping.")
            break

        # Elitism: carry top N individuals to next generation
        elite_idx = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i] for i in elite_idx]

        # Generate rest of population
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = population[random.choices(range(population_size), k=tournament_k)[0]]
            parent2 = population[random.choices(range(population_size), k=tournament_k)[0]]

            # Uniform crossover
            child = [
                parent1[0] if random.random() < 0.5 else parent2[0],
                int((parent1[1] + parent2[1]) / 2)
            ]

            # Adaptive mutation
            current_mutation_rate = mutation_rate * (1 - gen / generations)
            if random.random() < current_mutation_rate:
                child[0] *= 10 ** random.uniform(-0.1, 0.1)
                child[1] += random.randint(-50, 50)
                child[1] = max(100, min(child[1], 3000))

            new_population.append(child)

        population = new_population

    # Train final GA model with best individual
    best_C, best_iter = best_overall_individual
    print(f"\nGA FINAL MODEL -> C={best_C:.4f}, max_iter={best_iter}, Best CV Accuracy={best_overall_score:.4f}")
    final_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=best_C,
        max_iter=best_iter,
        random_state=RANDOM_STATE
    )
    final_model.fit(X_train, y_train)

    return final_model


# --- EXPERIMENT EXTENSION FUNCTION (Team-Specific Modeling) ---

def train_and_save_team_model(df, target_teams, model_name_prefix):
    """
    Trains and saves models using data filtered for a specific set of teams.
    This serves as the extension/improvement part of the project.
    """
    
    # Filter the DataFrame to include only matches where one of the target teams played
    df_filtered = df[
        (df["home_team"].isin(target_teams)) | 
        (df["away_team"].isin(target_teams))
    ].copy()

    # --- START OF TEAM-SPECIFIC ML PIPELINE ---
    team_list_str = f"{target_teams[0]} vs {target_teams[1]}" if len(target_teams) == 2 else ', '.join(target_teams)
    print(f"\n--- Running ML Pipeline Extension: {model_name_prefix} (Teams: {team_list_str}) ---")
    
    total_matches = len(df_filtered)
    print(f"Total filtered matches: {total_matches}")
    
    if total_matches < 100:
        print("WARNING: Dataset is small. Results are likely overfit and highly variable.")
        
    X = df_filtered[FEATURES]
    y = df_filtered["match_outcome"]
    
    # Check class distribution for the filtered data (can be different from global)
    print("\n--- Filtered Match Outcome Class Distribution ---")
    print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    print("------------------------------------------------")
    
    # Split (stratify is essential to maintain class balance in the small set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"scaler_{model_name_prefix}.pkl")


    # --- Train Logistic Regression (Simplified Model for Comparison) ---
    print("\n--- Training LogReg for Filtered Data ---")
    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)
    acc, f1, roc = evaluate_model(logreg, X_test_scaled, y_test)
    
    # Print metrics for comparison in presentation
    print(f"{model_name_prefix} LogReg Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    
    joblib.dump(logreg, f"{model_name_prefix}_logreg_model.pkl")
    joblib.dump({"acc": acc, "f1": f1, "roc": roc}, f"{model_name_prefix}_logreg_metrics.pkl")
    
    print(f"--- {model_name_prefix} Training Completed ---")


# --- MAIN TRAINING PIPELINE (Global Model) ---

def train_and_save_all():
    print("\n--- Starting ML Training Pipeline (Global Model) ---")

    df = load_data_and_create_features()
    X = df[FEATURES]
    y = df["match_outcome"]
    
    # Check Global Class Distribution
    print("\n--- Global Match Outcome Class Distribution ---")
    print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    print("---------------------------------------------")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # --- Training Logistic Regression (Baseline) ---
    print("\n--- Training Logistic Regression ---")
    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)
    acc, f1, roc = evaluate_model(logreg, X_test_scaled, y_test)
    print(f"Logistic Regression -> Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    joblib.dump(logreg, "logreg_model.pkl")
    joblib.dump({"acc": acc, "f1": f1, "roc": roc}, "logreg_metrics.pkl")

    # --- Training MLP Classifier ---
    print("\n--- Training MLP Classifier ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=RANDOM_STATE,
        verbose=True  # shows iteration progress
    )
    mlp.fit(X_train_scaled, y_train)
    acc, f1, roc = evaluate_model(mlp, X_test_scaled, y_test)
    print(f"MLP Classifier -> Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    joblib.dump(mlp, "mlp_model.pkl")
    joblib.dump({"acc": acc, "f1": f1, "roc": roc}, "mlp_metrics.pkl")

    # --- Training GA Optimized Model ---
    ga_model = train_ga_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )
    acc, f1, roc = evaluate_model(ga_model, X_test_scaled, y_test)
    # Print metrics for the presentation
    print(f"GA Model -> Accuracy: {acc:.4f}, F1 Score: {f1:.4f}") 
    joblib.dump(ga_model, "ga_model.pkl")
    joblib.dump({"acc": acc, "f1": f1, "roc": roc}, "ga_metrics.pkl")

    # --- Global Model Training Completed Successfully ---
    print("\n--- Global Model Training Completed Successfully ---\n")
    
    
    # --- CALL FOR TEAM-SPECIFIC MODEL EXTENSION ---
    # Using 'Arsenal' and 'Chelsea' as agreed upon for the extension argument
    target_teams_to_test = ["Arsenal", "Chelsea"]
    train_and_save_team_model(
        df,  # Use the full loaded DataFrame
        target_teams_to_test, 
        "ARS_CHE" # Prefix for saving files
    )
    # ---------------------------------------------
    
    print("\n--- Full Pipeline Execution Finished ---\n")


if __name__ == "__main__":
    train_and_save_all()