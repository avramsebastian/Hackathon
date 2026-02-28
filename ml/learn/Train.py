"""
ml/learn/Train.py
=================
Train the Random Forest collision-risk model on synthetic data.

Usage::

    python ml/learn/Train.py
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TrafficModelTrainer:
    """Trains a Random Forest to predict GO / STOP at intersections."""

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=25,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    def train(self, train_csv_path: str, model_save_path: str) -> None:
        """Load CSV, train, evaluate and serialise the model.

        Parameters
        ----------
        train_csv_path : str
            Path to the training CSV (with a ``label`` column).
        model_save_path : str
            Where to write the ``.pkl`` model file.
        """
        print(f"[Train] Loading data from '{os.path.basename(train_csv_path)}' …")
        df_train = pd.read_csv(train_csv_path)

        stop_count = len(df_train[df_train["label"] == 0])
        go_count = len(df_train[df_train["label"] == 1])
        print(f"[Train] GO: {go_count}  |  STOP: {stop_count}")

        X = df_train.drop("label", axis=1).values
        y = df_train["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42,
        )

        print(f"[Train] Training on {X.shape[1]} features …")
        self.model.fit(X_train, y_train)

        acc_train = self.model.score(X_train, y_train)
        acc_test = self.model.score(X_test, y_test)

        print("\n--- Performance Report ---")
        print(f"  Train accuracy: {acc_train * 100:.2f}%")
        print(f"  Test  accuracy: {acc_test * 100:.2f}%\n")

        joblib.dump(self.model, model_save_path)
        print(f"[Train] Model saved to '{os.path.basename(model_save_path)}'.")


if __name__ == "__main__":
    _ml_dir = os.path.abspath(os.path.dirname(__file__))
    _csv_path = os.path.join(_ml_dir, "..", "generated", "train_dataset.csv")
    _model_path = os.path.join(_ml_dir, "..", "generated", "traffic_model.pkl")

    trainer = TrafficModelTrainer()
    trainer.train(_csv_path, _model_path)