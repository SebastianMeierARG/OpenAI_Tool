from shiny import App, ui, render, reactive
import pandas as pd
import yaml
from src.data_generator import SyntheticDataLoader
from src.calculations import CreditRiskFormulas
from src.ai_engine import RiskReporter

# Load Config initially
with open("config.yaml", "r") as f:
    initial_config = yaml.safe_load(f)

# Initialize global modules
loader = SyntheticDataLoader()
calculator = CreditRiskFormulas(initial_config) # Note: calculator needs fresh config if params change
reporter = RiskReporter("config.yaml")

# Pre-generate data for the session (or could be reactive)
df_auc = loader.generate_auc_data()
df_calib = loader.generate_calibration_data()
df_scores = loader.generate_score_data()

app_ui = ui.page_fluid(
    ui.panel_title("Credit Risk Validation Dashboard"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Settings"),
            ui.input_numeric("alpha", "Significance Level (Alpha)", value=initial_config['thresholds']['significance_level_alpha'], step=0.01),
            ui.input_numeric("auc_tol", "AUC Tolerance", value=initial_config['thresholds']['auc_tolerance'], step=0.01),
            ui.hr(),
            ui.input_action_button("run_analysis", "Generate Report", class_="btn-primary")
        ),

        ui.navset_card_tab(
            ui.nav_panel("Validation Stats",
                ui.h4("AUC Trends"),
                ui.output_data_frame("auc_table"),
                ui.h4("Calibration Data (Chi-Square & Binomial inputs)"),
                ui.output_data_frame("calib_table"),
                ui.h4("Detailed Test Results"),
                ui.output_text_verbatim("stats_summary")
            ),
            ui.nav_panel("AI Report",
                ui.output_ui("report_markdown")
            )
        )
    )
)

def server(input, output, session):

    # Reactive calculator based on inputs
    @reactive.calc
    def get_calculator():
        # Update config dictionary with UI inputs
        current_config = initial_config.copy()
        current_config['thresholds']['significance_level_alpha'] = input.alpha()
        current_config['thresholds']['auc_tolerance'] = input.auc_tol()
        return CreditRiskFormulas(current_config)

    @render.data_frame
    def auc_table():
        return df_auc

    @render.data_frame
    def calib_table():
        return df_calib

    @render.text
    def stats_summary():
        calc = get_calculator()
        auc_stats = calc.evaluate_auc(df_auc)
        chi_stats = calc.calculate_chi_square(df_calib)
        ttest_stats = calc.calculate_ttest(df_scores)

        return (
            f"AUC Status: {auc_stats['Status']} (Diff: {auc_stats['Latest_AUC_Diff']:.4f})\n"
            f"Chi-Square P-Value: {chi_stats['P_Value']:.4f} ({chi_stats['Status']})\n"
            f"Score T-Test P-Value: {ttest_stats['P_Value']:.4f} ({ttest_stats['Status']})"
        )

    @render.ui
    @reactive.event(input.run_analysis)
    def report_markdown():
        calc = get_calculator()

        # Recalculate everything with current settings
        auc_stats = calc.evaluate_auc(df_auc)
        chi_stats = calc.calculate_chi_square(df_calib)
        binomial_results = calc.calculate_binomial_test(df_calib)
        jeffrey_results = calc.calculate_jeffrey_test(df_calib)
        ttest_stats = calc.calculate_ttest(df_scores)

        # Update Reporter config (hacky, ideally pass config to generate_report)
        # For now, we assume the reporter reads from file, but we want dynamic params.
        # The prompt uses {alpha} which we can override if we modify ai_engine to take params.
        # BUT, ai_engine reads config.yaml.
        # To support dynamic alpha in report, we should probably pass alpha to generate_report
        # or update the RiskReporter class.
        # Let's check RiskReporter.generate_report implementation.
        # It reads alpha from self.config.

        # We need to temporarily update reporter's config
        reporter.config['thresholds']['significance_level_alpha'] = input.alpha()

        with ui.Progress(min=1, max=15) as p:
            p.set(message="Querying AI Model...", detail="This may take a few seconds")
            report = reporter.generate_report(
                auc_stats, chi_stats, binomial_results, jeffrey_results, ttest_stats
            )

        return ui.markdown(report)

app = App(app_ui, server)
