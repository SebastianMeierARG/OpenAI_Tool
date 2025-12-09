import yaml
import os
import pandas as pd
from src.data_loader_module import get_data_loader
from src.calculations import CreditRiskFormulas
from src.pipeline import ValidationPipeline

def main():
    print("--- Credit Risk Model Validation Pipeline (Enterprise Edition) ---")

    # Load Config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return

    # 1. Load Data (Factory)
    print("Loading Data...")
    try:
        loader = get_data_loader("config.yaml")
        df_auc = loader.load_auc_data()
        df_calib = loader.load_calibration_data()
        df_scores = loader.load_score_data()
        print(f"Data Loaded using {loader.__class__.__name__}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

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

    # 3. Run Validation Pipeline
    print("Running Validation Pipeline (Generate -> Validate -> Evaluate)...")
    pipeline = ValidationPipeline("config.yaml")

    result = pipeline.run(
        auc_stats, chi_stats, binomial_results, jeffrey_results, ttest_stats
    )

    print(f"Final Status: {result['status']}")
    print(f"Trust Score: {result['score']}/10")
    print(f"Attempts: {result['attempts']}")

    # 4. Save Output
    output_file = "Credit_Risk_Report.md"
    with open(output_file, "w") as f:
        f.write(result['report'])

    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    main()
