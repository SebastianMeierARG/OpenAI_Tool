import pandas as pd
import numpy as np
from scipy import stats

class CreditRiskFormulas:
    """
    Performs statistical calculations for Credit Risk Model Validation.
    """
    def __init__(self, config):
        self.alpha = config['thresholds']['significance_level_alpha']
        self.auc_tolerance = config['thresholds']['auc_tolerance']
        self.jeffrey_priors = config['thresholds']['jeffrey_priors']

    def evaluate_auc(self, df_auc):
        """
        Evaluates AUC stability.
        """
        # Simple check: Is the latest AUC difference within tolerance?
        latest = df_auc.iloc[-1]
        diff = latest['AUC_Current'] - latest['AUC_Initial']
        status = "Accept" if abs(diff) <= self.auc_tolerance else "Reject"

        return {
            "Latest_AUC_Diff": diff,
            "Status": status,
            "Details": f"Diff {diff:.4f} vs Tolerance {self.auc_tolerance}"
        }

    def calculate_chi_square(self, df_calib):
        """
        Performs Pearson Chi-Square test for calibration.
        """
        # Expected defaults = N * PD
        expected_defaults = df_calib['N'] * df_calib['PD']
        observed_defaults = df_calib['Defaults']

        # Avoid division by zero or negative expected counts
        # This is a simplified implementation.
        # Chi2 = sum((Obs - Exp)^2 / Exp) + sum((NonDefault_Obs - NonDefault_Exp)^2 / NonDefault_Exp)
        # But often simpler version is used for just defaults if N is large.
        # Let's use scipy.stats.chisquare for goodness of fit if we have bins

        # More robust:
        obs = np.array([observed_defaults, df_calib['N'] - observed_defaults]).T
        exp = np.array([expected_defaults, df_calib['N'] - expected_defaults]).T

        # Summing up across all grades for a global fit test
        total_obs_def = observed_defaults.sum()
        total_exp_def = expected_defaults.sum()
        total_obs_non = (df_calib['N'] - observed_defaults).sum()
        total_exp_non = (df_calib['N'] - expected_defaults).sum()

        f_obs = [total_obs_def, total_obs_non]
        f_exp = [total_exp_def, total_exp_non]

        chi2, p_value = stats.chisquare(f_obs, f_exp)

        status = "Accept" if p_value > self.alpha else "Reject"

        return {
            "Chi2_Stat": chi2,
            "P_Value": p_value,
            "Status": status
        }

    def calculate_binomial_test(self, df_calib):
        """
        Performs Binomial test for each grade.
        """
        results = []
        for index, row in df_calib.iterrows():
            # Null hypothesis: p = PD
            # Alternative: p != PD
            p_val = stats.binomtest(int(row['Defaults']), int(row['N']), row['PD'], alternative='two-sided').pvalue
            status = "Accept" if p_val > self.alpha else "Reject"

            results.append({
                "Grade": row['Grade'],
                "Binomial_P_Value": p_val,
                "Status": status
            })

        return pd.DataFrame(results)

    def calculate_jeffrey_test(self, df_calib):
        """
        Calculates Jeffrey's Confidence Interval for Default Rate.
        """
        alpha_val = self.alpha
        results = []

        # Beta distribution quantile function
        # Lower bound: Beta(alpha/2; k + 0.5, n - k + 0.5)
        # Upper bound: Beta(1 - alpha/2; k + 0.5, n - k + 0.5)

        for index, row in df_calib.iterrows():
            k = row['Defaults']
            n = row['N']

            # Jeffrey's prior parameters are 0.5, 0.5
            a = k + 0.5
            b = n - k + 0.5

            lower = stats.beta.ppf(alpha_val / 2, a, b)
            upper = stats.beta.ppf(1 - alpha_val / 2, a, b)

            # Check if PD is within the interval
            pd_target = row['PD']
            status = "Accept" if lower <= pd_target <= upper else "Reject"

            results.append({
                "Grade": row['Grade'],
                "Jeffrey_Lower": lower,
                "Jeffrey_Upper": upper,
                "PD_Target": pd_target,
                "Status": status
            })

        return pd.DataFrame(results)

    def calculate_ttest(self, df_scores):
        """
        Performs T-test for score stability.
        """
        t_stat, p_val = stats.ttest_ind(df_scores['Score_Dev'], df_scores['Score_Curr'], equal_var=False)
        status = "Accept" if p_val > self.alpha else "Reject"

        return {
            "T_Stat": t_stat,
            "P_Value": p_val,
            "Status": status
        }