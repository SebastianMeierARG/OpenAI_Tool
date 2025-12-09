import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime, timedelta

class SyntheticDataLoader:
    """
    Generates synthetic data for Credit Risk Model Validation.
    """
    def __init__(self, seed=42):
        np.random.seed(seed)

    def load_auc_data(self):
        """
        Generates a time series of AUC values.
        """
        n_months = 12
        dates = [datetime.now() - timedelta(days=30 * i) for i in range(n_months)][::-1]
        dates = [d.strftime("%Y-%m-%d") for d in dates]

        # Simulate consistent AUC with some noise
        initial_auc = 0.75
        current_aucs = [initial_auc + np.random.normal(0, 0.02) for _ in range(n_months)]

        df = pd.DataFrame({
            "Date": dates,
            "AUC_Initial": [initial_auc] * n_months,
            "AUC_Current": current_aucs
        })
        return df

    def load_calibration_data(self):
        """
        Generates calibration data (PD vs Default Rate) for different rating grades.
        """
        grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        pds = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
        counts = [1000, 1500, 2000, 1500, 1000, 500, 200]

        data = []
        for g, pd_target, n in zip(grades, pds, counts):
            # Simulate defaults based on PD
            defaults = np.random.binomial(n, pd_target)
            data.append({
                "Grade": g,
                "PD": pd_target,
                "N": n,
                "Defaults": defaults,
                "Observed_DR": defaults / n
            })

        return pd.DataFrame(data)

    def load_score_data(self):
        """
        Generates synthetic credit scores for T-test (Development vs Current).
        """
        n_samples = 1000
        # Dev scores
        dev_scores = np.random.normal(600, 50, n_samples)
        # Current scores (shifted slightly)
        curr_scores = np.random.normal(595, 52, n_samples)

        return pd.DataFrame({
            "Score_Dev": dev_scores,
            "Score_Curr": curr_scores
        })

class CSVDataLoader:
    """
    Loads data from CSV files for Credit Risk Model Validation.
    """
    def __init__(self, csv_folder):
        self.csv_folder = csv_folder
        self._verify_files()

    def _verify_files(self):
        required_files = ["auc_data.csv", "calibration_data.csv", "score_data.csv"]
        missing = []
        for f in required_files:
            if not os.path.exists(os.path.join(self.csv_folder, f)):
                missing.append(f)

        if missing:
            raise FileNotFoundError(
                f"Missing required CSV files in {self.csv_folder}: {', '.join(missing)}. "
                "Please ensure all required files are present."
            )

    def load_auc_data(self):
        return pd.read_csv(os.path.join(self.csv_folder, "auc_data.csv"))

    def load_calibration_data(self):
        return pd.read_csv(os.path.join(self.csv_folder, "calibration_data.csv"))

    def load_score_data(self):
        return pd.read_csv(os.path.join(self.csv_folder, "score_data.csv"))

def get_data_loader(config_path="config.yaml"):
    """
    Factory to return the appropriate loader based on config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mode = config.get("data", {}).get("data_mode", "synthetic")

    if mode == "csv":
        csv_folder = config.get("data", {}).get("csv_folder", "./metrics_from_client")
        return CSVDataLoader(csv_folder)
    else:
        return SyntheticDataLoader()
