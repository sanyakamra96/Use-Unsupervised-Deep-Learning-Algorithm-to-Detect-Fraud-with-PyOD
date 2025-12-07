"""
Fraud Detection using PyOD AutoEncoder on Credit Card Transactions

2025 Fall - Advanced Artificial Intelligence (MSCS-633)
Hands-On Assignment 4: Use Unsupervised Deep Learning Algorithm to Detect Fraud with PyOD

Author: <Your Name>

This script:
- Loads the Kaggle credit card dataset (creditcard.csv)
- Preprocesses and scales the features
- Trains a PyOD AutoEncoder on mostly normal transactions
- Uses reconstruction error to detect fraud (outliers)
- Evaluates performance with common metrics

Make sure 'creditcard.csv' is in the same folder as this script, or update the path below.
"""
import kagglehub

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from pyod.models.auto_encoder import AutoEncoder


def load_data(csv_path: str = "creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card fraud dataset.

    Args:
        csv_path: Path to the creditcard.csv file.
    Returns:
        Pandas DataFrame with the loaded data.
    """
    kaggle_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("-------------Kaggle dataset downloaded to:", kaggle_path)

    kaggle_csv_path = os.path.join(kaggle_path, "creditcard.csv")
    if not os.path.exists(kaggle_csv_path):
        raise FileNotFoundError(
            f"'creditcard.csv' not found in Kaggle download directory: {kaggle_csv_path}"
        )

    return pd.read_csv(kaggle_csv_path)


def preprocess_data(df: pd.DataFrame):
    """
    Split the data into features and labels, scale the features,
    and create train/test splits.

    Strategy:
    - X: all columns except 'Class'
    - y: the 'Class' column (0 = normal, 1 = fraud)
    - Train mostly on normal samples to mimic unsupervised / anomaly detection

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Separate features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split with stratification to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # For unsupervised/anomaly detection, we typically train on mostly *normal* data.
    # Filter training data to include only normal transactions (Class = 0).
    normal_train_mask = (y_train == 0)
    X_train_normal = X_train[normal_train_mask]

    print(f"Original training samples: {len(X_train)}")
    print(f"Normal training samples used: {len(X_train_normal)}")
    print(f"Fraud samples in training (ignored for training): {sum(~normal_train_mask)}")

    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)  # test contains both classes

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_autoencoder_model(contamination: float = 0.001) -> AutoEncoder:
    """
    Create the PyOD AutoEncoder model.

    Args:
        contamination: Expected proportion of outliers in the data.
                       For credit card fraud, it is very low.
    Returns:
        Configured AutoEncoder model.
    """
    model = AutoEncoder(
    contamination=0.1,
    preprocessing=True,
    lr=1e-3,
    epoch_num=10,
    batch_size=32,
    optimizer_name='adam',
    device=None,
    random_state=42,
    use_compile=False,
    compile_mode='default',
    verbose=1,
    optimizer_params={'weight_decay': 1e-5},
    hidden_neuron_list=[64, 32],
    hidden_activation_name='relu',
    batch_norm=True,
    dropout_rate=0.2
    )

    return model


def evaluate_model(model: AutoEncoder, X_test_scaled, y_test):
    """
    Evaluate the AutoEncoder on test data.

    Args:
        model: Trained PyOD AutoEncoder model.
        X_test_scaled: Scaled features of the test set.
        y_test: True labels (0 = normal, 1 = fraud).

    Prints:
        Classification report, confusion matrix, ROC-AUC score.

    Also shows:
        ROC curve and histogram of anomaly scores.
    """
    # PyOD uses:
    #   .decision_scores_ for training scores
    #   .decision_function(X) for anomaly scores on new data
    #   .predict(X) for binary labels (0 = inlier, 1 = outlier)

    # Get binary predictions and scores
    y_pred = model.predict(X_test_scaled)          # 0 = normal, 1 = fraud (outlier)
    scores = model.decision_function(X_test_scaled)

    print("\n=== Classification Report (0 = normal, 1 = fraud) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Compute ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, scores)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("\nROC-AUC could not be computed (perhaps only one class present).")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, scores)
    plt.figure()
    plt.plot(fpr, tpr, label="AutoEncoder (Anomaly Score)")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection (AutoEncoder)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot histogram of anomaly scores
    plt.figure()
    plt.hist(scores[y_test == 0], bins=50, alpha=0.6, label="Normal (Class 0)")
    plt.hist(scores[y_test == 1], bins=50, alpha=0.6, label="Fraud (Class 1)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Distribution of Anomaly Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main entry point: load data, preprocess, train AutoEncoder, evaluate.
    """
    print("=== Fraud Detection with PyOD AutoEncoder ===")

    # 1. Load data
    df = load_data("creditcard.csv")
    print("Dataset loaded successfully.")
    print(df.head())
    print("\nClass distribution:")
    print(df["Class"].value_counts())

    # 2. Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)

    # 3. Build model
    contamination_rate = df["Class"].mean()  # approximate fraction of fraud
    print(f"\nEstimated contamination rate in dataset: {contamination_rate:.6f}")

    model = build_autoencoder_model(contamination=contamination_rate)

    # 4. Train model
    print("\n=== Training AutoEncoder model ===")
    model.fit(X_train_scaled)
    print("Model training completed.")

    # 5. Evaluate model
    evaluate_model(model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()
