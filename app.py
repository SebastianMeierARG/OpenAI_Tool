from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd
import data_loader
from rag_engine import RagEngine

# Load Data
df_overview, df_structure = data_loader.load_data()

# Initialize RAG Engine
rag = RagEngine()

app_ui = ui.page_navbar(
    ui.nav_panel("Risk Dashboard",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Filters"),
                ui.input_date_range(
                    "date_range",
                    "Select Date Range",
                    start=df_overview['Reference_Date'].min() if not df_overview.empty else None,
                    end=df_overview['Reference_Date'].max() if not df_overview.empty else None
                ),
                ui.input_select(
                    "grade_select",
                    "Select Grade (for ODR)",
                    choices=list(df_structure['Grade'].unique()) if not df_structure.empty else [],
                    selected=list(df_structure['Grade'].unique())[0] if not df_structure.empty else None
                )
            ),
            ui.layout_columns(
                ui.value_box(
                    "Current Gini",
                    ui.output_text("current_gini_val"),
                    showcase=None
                ),
                ui.value_box(
                    "Current ODR",
                    ui.output_text("current_odr_val"),
                    showcase=None
                )
            ),
            ui.card(
                ui.card_header("Gini Evolution"),
                output_widget("gini_plot")
            )
        )
    ),
    ui.nav_panel("Regulatory Chat",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Instructions"),
                ui.markdown("Ask questions about regulations or the current portfolio data. The AI uses the context from the dashboard.")
            ),
            ui.chat_ui("chat")
        )
    ),
    title="Credit Risk Dashboard"
)

def server(input, output, session):

    # --- Reactive Data ---
    @reactive.calc
    def filtered_overview():
        if df_overview.empty: return df_overview
        dr = input.date_range()
        if not dr: return df_overview # Handle init
        start_date = pd.to_datetime(dr[0])
        end_date = pd.to_datetime(dr[1])
        mask = (pd.to_datetime(df_overview['Reference_Date']) >= start_date) & \
               (pd.to_datetime(df_overview['Reference_Date']) <= end_date)
        return df_overview.loc[mask]

    @reactive.calc
    def current_metrics():
        df = filtered_overview()
        if df.empty:
            return {"gini": "N/A", "date": "N/A"}
        latest_row = df.iloc[-1]
        return {
            "gini": f"{latest_row['Gini']:.4f}",
            "date": str(latest_row['Reference_Date'])
        }

    @reactive.calc
    def current_odr():
        if df_structure.empty: return "N/A"
        grade = input.grade_select()
        if not grade: return "N/A"
        row = df_structure[df_structure['Grade'] == grade]
        if row.empty: return "N/A"
        val = row.iloc[0]['ODR_Calculated']
        return f"{val:.4f}"

    # --- UI Outputs ---
    @render.text
    def current_gini_val():
        return current_metrics()["gini"]

    @render.text
    def current_odr_val():
        return current_odr()

    @render_widget
    def gini_plot():
        df = filtered_overview()
        if df.empty:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Reference_Date'],
            y=df['Gini'],
            mode='lines+markers',
            name='Gini'
        ))
        fig.update_layout(
            title='Gini Evolution',
            xaxis_title='Date',
            yaxis_title='Gini',
            template='plotly_white'
        )
        return fig

    # --- Chat Logic ---
    chat = ui.Chat(id="chat", messages=[])

    @chat.on_user_submit
    async def _():
        user_input = chat.user_input()
        # Get context
        gini_info = current_metrics()
        odr_val = current_odr()
        context_str = f"Current Selected Gini: {gini_info['gini']} (Date: {gini_info['date']}). Selected Grade ODR: {odr_val}."

        # Append user message
        await chat.append_message({"role": "user", "content": user_input})

        # Get response
        response = rag.query(user_input, context_data=context_str)

        await chat.append_message({"role": "assistant", "content": response})

app = App(app_ui, server)
