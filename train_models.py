from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def build_pipeline(X, estimator):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
    return Pipeline([("preprocessing", prep), ("classifier", estimator)])

def run_experiment(name, X, y, class_names=None):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].copy()
    print("Records used:", len(X))
    print("Input features:", len(X.columns))
    print("Target distribution:\n", y.value_counts())

    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        print("Not enough observations in every class for reliable stratified evaluation.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    min_train_class = int(y_train.value_counts().min())
    folds = min(5, min_train_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4,
                                                class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
                                                class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    pipes = {k: build_pipeline(X_train, v) for k, v in estimators.items()}
    voting = VotingClassifier(
        estimators=[
            ("logistic", estimators["Logistic Regression"]),
            ("forest", estimators["Random Forest"]),
            ("boosting", estimators["Gradient Boosting"])
        ],
        voting="soft"
    )
    pipes["Combined Voting Model"] = build_pipeline(X_train, voting)

    rows, fitted = [], {}
    for model_name, model in pipes.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[model_name] = model
        average = "weighted"
        rows.append({
            "Dataset": name,
            "Model": model_name,
            "CV accuracy average": cv_acc.mean(),
            "CV accuracy variation": cv_acc.std(),
            "Test accuracy": accuracy_score(y_test, pred),
            "Balanced accuracy": balanced_accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average=average, zero_division=0),
            "Recall": recall_score(y_test, pred, average=average, zero_division=0),
            "F1 score": f1_score(y_test, pred, average=average, zero_division=0)
        })
        print(f"\n{model_name}")
        print(f"  Cross-validation accuracy: {cv_acc.mean():.1%} (variation {cv_acc.std():.1%})")
        print(f"  Held-out test accuracy: {accuracy_score(y_test, pred):.1%}")

    result = pd.DataFrame(rows).sort_values("CV accuracy average", ascending=False)
    safe = name.lower().replace(" ", "_")
    result.to_csv(OUT / f"{safe}_model_comparison.csv", index=False)

    best_name = result.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    joblib.dump(best_model, OUT / f"{safe}_best_model.joblib")

    print("\nSelected model:", best_name)
    print("\nDetailed test report:")
    print(classification_report(y_test, best_pred, zero_division=0))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, xticks_rotation=30, values_format="d"
    )
    plt.title(f"{name} - {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / f"{safe}_confusion_matrix.png", dpi=220)
    plt.close()

    plot = result.sort_values("Test accuracy")
    plt.figure(figsize=(9, 5))
    plt.barh(plot["Model"], plot["Test accuracy"])
    plt.xlabel("Test accuracy")
    plt.xlim(0, 1)
    plt.title(f"{name}: Model accuracy comparison")
    plt.tight_layout()
    plt.savefig(OUT / f"{safe}_accuracy_comparison.png", dpi=220)
    plt.close()
    return result

def main():
    summaries = []

    survey_path = BASE / "Untitled form.csv"
    if survey_path.exists():
        df = pd.read_csv(survey_path)
        target_col = "Before seeing the product/offer, had you already planned to buy it?"
        if target_col in df.columns:
            y_raw = df[target_col].astype(str).str.strip()

            mapping = {
                "Yes": 0,
                "No": 1,
                "I was considering buying something similar": 1
            }
            y = y_raw.map(mapping)
            X = df.drop(columns=[target_col, "Timestamp"], errors="ignore")
            r = run_experiment("Google Forms Survey", X, y)
            if r is not None: summaries.append(r)
        else:
            print("Survey target question was not found. Check the CSV heading.")

    social_path = BASE / "social_media_buying.csv"
    if social_path.exists():
        df = pd.read_csv(social_path)
        if "impulse_purchase" in df.columns:
            y = pd.to_numeric(df["impulse_purchase"], errors="coerce")
            X = df.drop(columns=["impulse_purchase"])
            r = run_experiment("Social Media Buying", X, y)
            if r is not None: summaries.append(r)

    psych_path = BASE / "Cleaned dataset.xlsx"
    if psych_path.exists():
        df = pd.read_excel(psych_path)
        items = ["IB1", "IB2", "IB3", "IB4"]
        if all(c in df.columns for c in items):
            ib_mean = df[items].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            y = pd.Series(np.where(ib_mean >= 4, 1, 0), index=df.index)

            X = df.drop(columns=items)
            r = run_experiment("Psychological Drivers", X, y)
            if r is not None: summaries.append(r)

    if summaries:
        all_results = pd.concat(summaries, ignore_index=True)
        all_results.to_csv(OUT / "ALL_MODEL_RESULTS.csv", index=False)
        print("\nAll experiments completed.")
        print("Score tables, graphs, confusion matrices and saved models in results folder.")
    else:
        print("No experiment ran. Check that all three data files are beside this script.")

if __name__ == "__main__":
    main()
