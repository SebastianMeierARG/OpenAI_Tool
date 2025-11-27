# Credit Risk Reporting AI Application

A modular, production-ready AI application for Credit Risk Reporting. This tool generates synthetic credit risk data, performs statistical validation (AUC, Chi-Square, Binomial Tests, Jeffrey's Test, T-Test), and uses an LLM to generate a comprehensive Model Validation Report.

## Features

-   **Modular Architecture**: Logic separated into Data Generation, Calculations, and AI Reporting.
-   **Statistical Validation**: automated calculation of key metrics (AUC trends, Calibration tests, Population Stability).
-   **AI-Powered Reporting**: Generates Markdown reports using OpenAI (GPT-4o).
-   **Multiple Interfaces**:
    -   **CLI**: Quick batch execution.
    -   **Web Dashboard**: Interactive Shiny for Python UI.
    -   **Jupyter Notebook**: Step-by-step manual verification.
-   **Configurable**: All thresholds and prompts are managed in `config.yaml`.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **API Key Setup**:
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Open `.env` and paste your OpenAI API Key:
        ```
        OPENAI_API_KEY=sk-your-key-here
        ```

2.  **Adjust Settings**:
    -   Open `config.yaml` to modify:
        -   **Thresholds**: Significance level (Alpha), AUC tolerance.
        -   **AI Settings**: Model name, temperature.
        -   **Prompts**: System persona and report instructions.

## Usage

### 1. Command Line Interface (CLI)
Generates data, calculates stats, and saves a report to `Credit_Risk_Report.md`.

```bash
python main.py
```

### 2. Web Dashboard (Shiny App)
Launches an interactive web UI to visualize data, tweak parameters dynamically, and generate reports.

```bash
shiny run app.py
```
*Open the provided local URL (e.g., http://127.0.0.1:8000) in your browser.*

### 3. Manual Testing (Jupyter Notebook)
Step through the logic interactively.

1.  Start Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `notebooks/manual_test.ipynb`.
3.  Run cells sequentially to inspect intermediate DataFrames and results.

## Project Structure

```
├── src/                # Core logic modules
│   ├── data_generator.py
│   ├── calculations.py
│   └── ai_engine.py
├── data/               # Input/Output directory
├── notebooks/          # Jupyter notebooks
├── config.yaml         # Configuration file
├── main.py             # CLI entry point
├── app.py              # Shiny web application
├── requirements.txt    # Dependencies
└── .env                # API Keys (not committed)
```
