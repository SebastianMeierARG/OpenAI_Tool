from shiny import App, ui, render, reactive
import pandas as pd
import yaml
import asyncio
from src.data_loader_module import get_data_loader
from src.calculations import CreditRiskFormulas
from src.pipeline import ValidationPipeline
from src.rag_engine import RagEngine

# Load Config initially
with open("config.yaml", "r") as f:
    initial_config = yaml.safe_load(f)

# Initialize global modules
# Pipeline is heavy, initialize when needed or keep one instance if stateless enough (it is)
pipeline = ValidationPipeline("config.yaml")
rag = RagEngine("config.yaml")

# Data Loader (Using Factory)
# We might want to reload this if user changes config, but for now init at start
data_loader = get_data_loader("config.yaml")

def load_all_data():
    try:
        df_auc = data_loader.load_auc_data()
        df_calib = data_loader.load_calibration_data()
        df_scores = data_loader.load_score_data()
        return df_auc, df_calib, df_scores, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), str(e)

df_auc_init, df_calib_init, df_scores_init, load_error = load_all_data()

app_ui = ui.page_fluid(
    ui.panel_title("Credit Risk Validation Dashboard (Enterprise)"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Settings"),
            ui.input_select("data_mode", "Data Source",
                            choices={"synthetic": "Synthetic", "csv": "CSV Folder"},
                            selected=initial_config['data'].get('data_mode', 'synthetic')),
            ui.input_action_button("reload_data", "Reload Data"),
            ui.hr(),
            ui.input_numeric("alpha", "Significance Level (Alpha)", value=initial_config['thresholds']['significance_level_alpha'], step=0.01),
            ui.input_numeric("auc_tol", "AUC Tolerance", value=initial_config['thresholds']['auc_tolerance'], step=0.01),
            ui.hr(),
            ui.input_action_button("run_analysis", "Generate Verified Report", class_="btn-primary"),
            ui.hr(),
            ui.h4("Knowledge Base"),
            ui.input_action_button("ingest_docs", "Ingest Documents"),
            ui.output_text("ingest_status")
        ),

        ui.navset_card_tab(
            ui.nav_panel("Validation Stats",
                ui.output_text_verbatim("data_status"),
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
            ),
            ui.nav_panel("Regulatory Chat",
                ui.chat_ui("chat")
            )
        )
    )
)

def server(input, output, session):

    # Reactive Data Containers
    data = reactive.Value((df_auc_init, df_calib_init, df_scores_init))
    data_err = reactive.Value(load_error)
    val_result = reactive.Value(None)
    ingest_msg = reactive.Value("")

    @reactive.effect
    @reactive.event(input.reload_data)
    def _():
        # Quick hack: Update config file in memory or just instantiate loader with new mode?
        # The factory reads from file. So we need to ensure file is consistent or modify factory to take params.
        # Ideally we shouldn't modify config.yaml at runtime for this, but for the MVP:
        # We will assume user changes config.yaml manually OR we pass args to factory.
        # But get_data_loader takes config_path.
        # Let's override the loader based on input.
        mode = input.data_mode()
        # We need a way to instantiate loader directly
        from src.data_loader_module import CSVDataLoader, SyntheticDataLoader

        try:
            if mode == "csv":
                loader = CSVDataLoader(initial_config['data']['csv_folder'])
            else:
                loader = SyntheticDataLoader()

            d_auc = loader.load_auc_data()
            d_calib = loader.load_calibration_data()
            d_scores = loader.load_score_data()
            data.set((d_auc, d_calib, d_scores))
            data_err.set(None)
        except Exception as e:
            data_err.set(str(e))
            data.set((pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))

    @reactive.calc
    def get_calculator():
        # Update config dictionary with UI inputs
        current_config = initial_config.copy()
        current_config['thresholds']['significance_level_alpha'] = input.alpha()
        current_config['thresholds']['auc_tolerance'] = input.auc_tol()
        return CreditRiskFormulas(current_config)

    @render.text
    def data_status():
        err = data_err()
        if err:
            return f"DATA LOAD ERROR: {err}"
        return "Data Loaded Successfully."

    @render.data_frame
    def auc_table():
        return data()[0]

    @render.data_frame
    def calib_table():
        return data()[1]

    @render.text
    def stats_summary():
        df_auc, df_calib, df_scores = data()
        if df_auc.empty: return "No Data"

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
        df_auc, df_calib, df_scores = data()
        if df_auc.empty: return

        calc = get_calculator()

        # Recalculate everything with current settings
        auc_stats = calc.evaluate_auc(df_auc)
        chi_stats = calc.calculate_chi_square(df_calib)
        binomial_results = calc.calculate_binomial_test(df_calib)
        jeffrey_results = calc.calculate_jeffrey_test(df_calib)
        ttest_stats = calc.calculate_ttest(df_scores)

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

    # --- RAG Logic ---

    @reactive.effect
    @reactive.event(input.ingest_docs)
    def _():
        with ui.Progress(min=1, max=5) as p:
            p.set(message="Ingesting Documents...", detail="Parsing PDFs and Indexing...")
            msg = rag.ingest_documents()
            ingest_msg.set(msg)

    @render.text
    def ingest_status():
        return ingest_msg()

    chat = ui.Chat(id="chat", messages=[])

    @chat.on_user_submit
    async def _():
        user_input = chat.user_input()

        # Get Context Stats
        stats_text = stats_summary() # Get summary string

        await chat.append_message({"role": "user", "content": user_input})

        # Query RAG
        # Note: This is blocking. In production, run in thread/async if possible.
        # RagEngine uses synchronous OpenAI calls mostly.
        response = rag.query_knowledge_base(user_input, stats_context=stats_text)

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
