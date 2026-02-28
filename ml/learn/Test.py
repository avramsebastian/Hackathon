"""
ml/learn/Test.py
================
Evaluate the trained model on the validation set and write a human-readable
report to ``ml/generated/rezultate_test.txt``.

Usage::

    python ml/learn/Test.py
"""

import pandas as pd
import joblib
import os


def test_saved_model(
    val_csv_path: str,
    model_path: str,
    output_file: str,
) -> None:
    """Load the model, score it on validation data and write a report.

    Parameters
    ----------
    val_csv_path : str
        Path to the validation CSV.
    model_path : str
        Path to the serialised ``.pkl`` model.
    output_file : str
        Where to write the human-readable report.
    """
    print(f"[Test] Loading model from '{model_path}' …")

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: model not found — run Train.py first.")
        return

    print(f"[Test] Loading validation data from '{val_csv_path}' …")
    df_val = pd.read_csv(val_csv_path)

    X_val = df_val.drop("label", axis=1).values
    y_val = df_val["label"].values

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write("=" * 50 + "\n")
        fh.write("         V2X AI MODEL TEST REPORT\n")
        fh.write("=" * 50 + "\n\n")

        # 1. Overall accuracy
        accuracy = model.score(X_val, y_val)
        msg = f"[Test] Validation accuracy: {accuracy * 100:.2f}%\n"
        print(msg)
        fh.write(msg + "\n")

        # 2. Per-scenario confidence breakdown (30 samples)
        fh.write("--- Confidence breakdown (30 scenarios) ---\n")
        for i in range(min(30, len(y_val))):
            features = X_val[i].reshape(1, -1)
            true_label = y_val[i]

            probabilities = model.predict_proba(features)[0]
            prob_go = probabilities[1]

            true_str = "GO" if true_label == 1 else "STOP"
            pred_str = "GO" if prob_go > 0.5 else "STOP"

            fh.write(f"\nScenario {i + 1} (true: {true_str}):\n")
            fh.write(f"  GO  confidence: {prob_go * 100:.1f}%\n")
            fh.write(f"  STOP confidence: {(1 - prob_go) * 100:.1f}%\n")
            fh.write(f"  Model decision: {pred_str}\n")

    print(f"\nDone — results saved to '{output_file}'.")


if __name__ == "__main__":
    _ml_dir = os.path.abspath(os.path.dirname(__file__))
    _csv_path = os.path.join(_ml_dir, "..", "generated", "val_dataset.csv")
    _model_path = os.path.join(_ml_dir, "..", "generated", "traffic_model.pkl")
    _out_path = os.path.join(_ml_dir, "..", "generated", "rezultate_test.txt")

    test_saved_model(_csv_path, _model_path, _out_path)