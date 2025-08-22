import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

experiment_base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
if not experiment_base_name:
    raise ValueError("Environment variable MLFLOW_EXPERIMENT_NAME is not set.")

experiment_name = f"train/{experiment_base_name}"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

# Create or get experiment
mlflow.set_experiment(experiment_name)

# Load dataset
df = pd.read_csv("data/train.csv")

# Basic cleaning
df = df.dropna(subset=["Age", "Fare", "Embarked"])
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

# Preprocessing
numeric_features = ["Age", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Model pipeline
pipeline = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ))
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Titanic") as run:
    # Train
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log model and register
    model_name = "Titanic_RandomForest_Model"
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=model_name  # ✅ Registers into MLflow Registry
    )
    
    mlflow.sklearn.log_model(
    pipeline,
    "model",
    input_example=X.iloc[:1],
    registered_model_name="Titanic_RandomForest_Model"
)

    print(f"✅ Model trained, logged, and registered in MLflow | Run ID={run.info.run_id}")
    print(f"Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

mlflow.sklearn.log_model(
    pipeline,
    "model",
    input_example=X.iloc[:1],
    registered_model_name="Titanic_RandomForest_Model"
)
