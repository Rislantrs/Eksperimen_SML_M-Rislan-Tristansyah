import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import os

# Load dataset hasil preprocessing (jalur relatif terhadap file script ini)
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "social_media_mental_health_advanced_cleaned.csv")
TARGET_COLUMN = "depression_label"

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

# MLflow akan otomatis mengatur tracking saat dijalankan via 'mlflow run'
# mlflow.set_tracking_uri("file:./mlruns")
# mlflow.set_experiment("Basic_Model_Experiment")

# Menggunakan autolog sesuai kriteria Basic
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Simpan run_id untuk keperluan CI/CD (Docker Build)
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)