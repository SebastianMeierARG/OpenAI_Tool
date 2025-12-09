from shiny import App, ui, render, reactive
import pandas as pd
import yaml
import os
from src.data_loader_module import get_data_loader
from src.calculations import CreditRiskFormulas
from src.pipeline import ValidationPipeline
from src.rag_engine import RagEngine

# Load Config initially
with open("config.yaml", "r") as f:
    initial_config = yaml.safe_load(f)

# Initialize global modules
pipeline = ValidationPipeline("config.yaml")
rag = RagEngine("config.yaml")

# Data Loader
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
    # Custom CSS and Theme
    ui.include_css("www/custom.css"),

    ui.layout_sidebar(
        ui.sidebar(
            # Logo Container
            ui.div(
                ui.img(src="logo.png", alt="Logo"), # User needs to put logo.png in www/
                class_="logo-container"
            ),
            ui.h4("Configuration", class_="mb-3"),

            # Accordion for settings
            ui.accordion(
                ui.accordion_panel("Data Source",
                    ui.input_select("data_mode", "Source Type",
                                    choices={"synthetic": "Synthetic", "csv": "CSV Folder"},
                                    selected=initial_config['data'].get('data_mode', 'synthetic')),
                    ui.input_action_button("reload_data", "Reload Data", class_="btn-sm w-100")
                ),
                ui.accordion_panel("Model Thresholds",
                    ui.input_numeric("alpha", "Significance (Alpha)", value=initial_config['thresholds']['significance_level_alpha'], step=0.01),
                    ui.input_numeric("auc_tol", "AUC Tolerance", value=initial_config['thresholds']['auc_tolerance'], step=0.01)
                ),
                ui.accordion_panel("Knowledge Base",
                    ui.input_action_button("ingest_docs", "Ingest Documents", class_="btn-sm w-100"),
                    ui.output_text("ingest_status", inline=True)
                ),
                open=True
            ),
            ui.hr(),
            ui.input_action_button("run_analysis", "Generate Verified Report", class_="btn-primary w-100")
        ),

        ui.navset_card_underline(
            ui.nav_panel("Validation Stats",
                ui.div(
                    ui.output_text_verbatim("data_status"),
                    class_="alert alert-info"
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("AUC Trends"),
                        ui.output_data_frame("auc_table")
                    ),
                    ui.card(
                        ui.card_header("Calibration Data"),
                        ui.output_data_frame("calib_table")
                    )
                ),
                ui.card(
                    ui.card_header("Test Summary"),
                    ui.output_text_verbatim("stats_summary")
                )
            ),
            ui.nav_panel("Verified Report",
                ui.layout_columns(
                    ui.value_box("Status", ui.output_text("ver_status"), theme="value-box-neutral"),
                    ui.value_box("Trust Score", ui.output_text("trust_score"), theme="value-box-neutral"),
                    ui.value_box("Attempts", ui.output_text("attempts_count"), theme="value-box-neutral"),
                ),
                ui.hr(),
                ui.card(
                    ui.output_ui("report_markdown")
                )
            ),
            ui.nav_panel("Regulatory Chat",
                ui.div(
                    ui.chat_ui("chat"),
                    class_="shiny-chat-container"
                )
            )
        )
    ),
    title="Enterprise Credit Risk Engine"
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
        mode = input.data_mode()
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
        current_config = initial_config.copy()
        current_config['thresholds']['significance_level_alpha'] = input.alpha()
        current_config['thresholds']['auc_tolerance'] = input.auc_tol()
        return CreditRiskFormulas(current_config)

    @render.text
    def data_status():
        err = data_err()
        if err: return f"ERROR: {err}"
        return f"Loaded: {input.data_mode().capitalize()} Data"

    @render.data_frame
    def auc_table():
        return data()[0]

    @render.data_frame
    def calib_table():
        return data()[1]

    @render.text
    def stats_summary():
        df_auc, df_calib, df_scores = data()
        if df_auc.empty: return "No Data Loaded"

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
        auc_stats = calc.evaluate_auc(df_auc)
        chi_stats = calc.calculate_chi_square(df_calib)
        binomial_results = calc.calculate_binomial_test(df_calib)
        jeffrey_results = calc.calculate_jeffrey_test(df_calib)
        ttest_stats = calc.calculate_ttest(df_scores)

        pipeline.config['thresholds']['significance_level_alpha'] = input.alpha()

        with ui.Progress(min=1, max=15) as p:
            p.set(message="Running Verification Pipeline...", detail="This process ensures no hallucinations...")
            res = pipeline.run(
                auc_stats, chi_stats, binomial_results, jeffrey_results, ttest_stats
            )
            val_result.set(res)

    # Dynamic Color for Value Boxes
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
        return ui.div(
            ui.h5("Ready to Generate"),
            ui.p("Click the button in the sidebar to run the full validation pipeline."),
            style="text-align: center; color: #6c757d; padding: 40px;"
        )

    # --- RAG Logic ---
    @reactive.effect
    @reactive.event(input.ingest_docs)
    def _():
        with ui.Progress(min=1, max=5) as p:
            p.set(message="Indexing Knowledge Base...", detail="Reading PDFs from /documents...")
            msg = rag.ingest_documents()
            ingest_msg.set(msg)

    @render.text
    def ingest_status():
        return ingest_msg()

    chat = ui.Chat(id="chat", messages=[])

    @chat.on_user_submit
    async def _():
        user_input = chat.user_input()
        stats_text = stats_summary()
        await chat.append_message({"role": "user", "content": user_input})
        response = rag.query_knowledge_base(user_input, stats_context=stats_text)
        await chat.append_message({"role": "assistant", "content": response})

app = App(app_ui, server)
