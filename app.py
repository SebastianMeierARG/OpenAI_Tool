from shiny import App, ui, render, reactive
import pandas as pd
import yaml
from src.data_generator import SyntheticDataLoader
from src.calculations import CreditRiskFormulas
from src.pipeline import ValidationPipeline

# Load Config initially
with open("config.yaml", "r") as f:
    initial_config = yaml.safe_load(f)

# Initialize global modules
loader = SyntheticDataLoader()
# Pipeline is heavy, initialize when needed or keep one instance if stateless enough (it is)
pipeline = ValidationPipeline("config.yaml")

# Pre-generate data for the session
df_auc = loader.generate_auc_data()
df_calib = loader.generate_calibration_data()
df_scores = loader.generate_score_data()

app_ui = ui.page_fluid(
    ui.panel_title("Credit Risk Validation Dashboard (Enterprise)"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Settings"),
            ui.input_numeric("alpha", "Significance Level (Alpha)", value=initial_config['thresholds']['significance_level_alpha'], step=0.01),
            ui.input_numeric("auc_tol", "AUC Tolerance", value=initial_config['thresholds']['auc_tolerance'], step=0.01),
            ui.hr(),
            ui.input_action_button("run_analysis", "Generate Verified Report", class_="btn-primary")
        ),

        ui.navset_card_tab(
            ui.nav_panel("Validation Stats",
                ui.h4("AUC Trends"),
                ui.output_data_frame("auc_table"),
                ui.h4("Calibration Data"),
                ui.output_data_frame("calib_table"),
                ui.h4("Detailed Test Results"),
                ui.output_text_verbatim("stats_summary")
            ),
            ui.nav_panel("Verified Report",
                ui.layout_columns(
                    ui.value_box("Status", ui.output_text("ver_status"), theme="bg-gradient-blue-purple"),
                    ui.value_box("Trust Score", ui.output_text("trust_score"), theme="bg-gradient-blue-purple"),
                    ui.value_box("Attempts", ui.output_text("attempts_count"), theme="bg-gradient-blue-purple"),
                ),
                ui.hr(),
                ui.output_ui("report_markdown")
            )
        )
    )
)

def server(input, output, session):

    # Store report result
    val_result = reactive.Value(None)

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

    @reactive.effect
    @reactive.event(input.run_analysis)
    def _():
        calc = get_calculator()

        # Recalculate everything with current settings
        auc_stats = calc.evaluate_auc(df_auc)
        chi_stats = calc.calculate_chi_square(df_calib)
        binomial_results = calc.calculate_binomial_test(df_calib)
        jeffrey_results = calc.calculate_jeffrey_test(df_calib)
        ttest_stats = calc.calculate_ttest(df_scores)

        # Update config for pipeline (alpha specifically)
        # Note: Pipeline reads from file or object.
        # Ideally we pass alpha to run().
        # But pipeline uses self.config loaded from file.
        # We need to hack the pipeline config or allow run() to accept overrides.
        # Let's check pipeline.run() implementation. It uses self.config for alpha.
        # We should update pipeline instance config.
        pipeline.config['thresholds']['significance_level_alpha'] = input.alpha()

        with ui.Progress(min=1, max=15) as p:
            p.set(message="Running Verification Pipeline...", detail="Generating, Validating, Evaluating...")
            res = pipeline.run(
                auc_stats, chi_stats, binomial_results, jeffrey_results, ttest_stats
            )
            val_result.set(res)

    @render.text
    def ver_status():
        res = val_result()
        return res['status'] if res else "Pending"

    @render.text
    def trust_score():
        res = val_result()
        return f"{res['score']}/10" if res else "-"

    @render.text
    def attempts_count():
        res = val_result()
        return str(res['attempts']) if res else "-"

    @render.ui
    def report_markdown():
        res = val_result()
        if res:
            return ui.markdown(res['report'])
        return ui.p("Click 'Generate Verified Report' to start.")

app = App(app_ui, server)
