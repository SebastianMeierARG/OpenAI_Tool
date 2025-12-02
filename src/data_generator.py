import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SyntheticDataLoader:
    """
    Generates synthetic data for Credit Risk Model Validation.
    """
    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_auc_data(self, n_months=12):
        """
        Generates a time series of AUC values.
        """
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

    def generate_calibration_data(self):
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

    def generate_score_data(self, n_samples=1000):
        """
        Generates synthetic credit scores for T-test (Development vs Current).
        """
        # Dev scores
        dev_scores = np.random.normal(600, 50, n_samples)
        # Current scores (shifted slightly)
        curr_scores = np.random.normal(595, 52, n_samples)

        return pd.DataFrame({
            "Score_Dev": dev_scores,
            "Score_Curr": curr_scores
        })
