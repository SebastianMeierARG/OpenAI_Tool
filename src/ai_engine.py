import os
import json
import yaml
from openai import OpenAI
from dotenv import load_dotenv

class RiskReporter:
    """
    Interacts with OpenAI to generate the Credit Risk Report.
    """
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # For safety, we can allow initialization without key but fail on generate
            print("Warning: OPENAI_API_KEY not found in environment.")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_report(self, auc_stats, chi_stats, binomial_df, jeffrey_df, ttest_stats):
        """
        Compiles the stats and requests a report from LLM.
        """
        if not self.client:
            return "Error: OpenAI API Key is missing. Cannot generate report."

        # Prepare Context Data
        context = {
            "AUC_Analysis": auc_stats,
            "Chi_Square_Test": chi_stats,
            "Binomial_Test_Results": binomial_df.to_dict(orient="records"),
            "Jeffrey_Test_Results": jeffrey_df.to_dict(orient="records"),
            "Population_Stability_T_Test": ttest_stats
        }

        context_str = json.dumps(context, indent=2)

        # Prepare Prompt
        system_persona = self.config['prompts']['system_persona']
        report_instruction = self.config['prompts']['report_instruction']
        alpha = self.config['thresholds']['significance_level_alpha']

        user_prompt = report_instruction.format(context_data=context_str, alpha=alpha)

        model = self.config['ai_settings']['model']
        temp = self.config['ai_settings']['temperature']
        max_tok = self.config['ai_settings']['max_tokens']

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating report: {str(e)}"
