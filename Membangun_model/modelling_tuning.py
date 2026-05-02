import json
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


import os

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "social_media_mental_health_advanced_cleaned.csv")
TARGET_COLUMN = "depression_label"

dagshub.init(
    repo_owner="Rislantrs",
    repo_name="Eksperimen_SML_Mental_Health",
    mlflow=True
)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

mlflow.set_experiment("Advance_DagsHub_Model_Experiment")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

base_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1
)

with mlflow.start_run(run_name="RandomForest_DagsHub_ManualLogging"):
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Manual logging parameters
    mlflow.log_params(grid_search.best_params_)

    # Manual logging metrics utama
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Manual logging metrics tambahan
    mlflow.log_metric("train_score", best_model.score(X_train, y_train))
    mlflow.log_metric("test_score", best_model.score(X_test, y_test))

    # Simpan classification report sebagai artifact tambahan
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Simpan confusion matrix sebagai artifact tambahan
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv("confusion_matrix.csv", index=False)

    mlflow.log_artifact("classification_report.json")
    mlflow.log_artifact("confusion_matrix.csv")

    # Manual logging model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)