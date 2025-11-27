import yaml
import os
from src.data_generator import SyntheticDataLoader
from src.calculations import CreditRiskFormulas
from src.ai_engine import RiskReporter

def main():
    print("--- Credit Risk Model Validation Pipeline ---")

    # Load Config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return

    # 1. Generate Data
    print("Generating Synthetic Data...")
    loader = SyntheticDataLoader()
    df_auc = loader.generate_auc_data()
    df_calib = loader.generate_calibration_data()
    df_scores = loader.generate_score_data()

    # 2. Calculate Stats
    print("Calculating Statistics...")
    calculator = CreditRiskFormulas(config)

    auc_stats = calculator.evaluate_auc(df_auc)
    chi_stats = calculator.calculate_chi_square(df_calib)
    binomial_results = calculator.calculate_binomial_test(df_calib)
    jeffrey_results = calculator.calculate_jeffrey_test(df_calib)
    ttest_stats = calculator.calculate_ttest(df_scores)

    print(f"AUC Status: {auc_stats['Status']}")
    print(f"Chi-Square Status: {chi_stats['Status']}")

    # 3. Generate AI Report
    print("Generating AI Report...")
    reporter = RiskReporter("config.yaml")
    report_content = reporter.generate_report(
        auc_stats, chi_stats, binomial_results, jeffrey_results, ttest_stats
    )

    # 4. Save Output
    output_file = "Credit_Risk_Report.md"
    with open(output_file, "w") as f:
        f.write(report_content)

    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    main()
