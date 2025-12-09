import os
import json
import yaml
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from dotenv import load_dotenv

class ModelEvaluator:
    """
    LLM-as-a-Judge to evaluate the generated report.
    """
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        # Setup Jinja2 for evaluator prompt
        self.env = Environment(loader=FileSystemLoader("templates"))
        self.template = self.env.get_template("evaluator_prompt.j2")

    def evaluate_report(self, generated_report, context_data):
        """
        Evaluates the report using an LLM.
        Returns: JSON object with score, pass/fail, and reasoning.
        """
        if not self.client:
            return {"score": 0, "pass": False, "reasoning": "OpenAI API Key missing."}

        # Render the prompt
        prompt = self.template.render(
            context_data=context_data,
            generated_report=generated_report
        )

        model = self.config['ai_settings']['model']
        # We might want to use a cheaper model for evaluation, but config says model: gpt-4o.
        # Let's stick to the config model or hardcode a "judge" model if specified.
        # For now, using the same model.

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an automated quality assurance system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0, # Deterministic for evaluation
                response_format={"type": "json_object"} # Ensure JSON output
            )

            result_json = json.loads(response.choices[0].message.content)

            # Enforce pass threshold from config
            min_score = self.config['thresholds'].get('min_faithfulness_score', 8)
            result_json['pass'] = result_json.get('score', 0) >= min_score

            return result_json

        except Exception as e:
            return {
                "score": 0,
                "pass": False,
                "reasoning": f"Evaluation failed: {str(e)}"
            }
