import json
import logging
import datetime
import hashlib
import yaml
import os
from jinja2 import Environment, FileSystemLoader
from src.ai_engine import RiskReporter
from src.validator import OutputValidator
from src.evaluator import ModelEvaluator

# Setup Logging
logging.basicConfig(
    filename='audit_log.json',
    level=logging.INFO,
    format='%(message)s'
)

class ValidationPipeline:
    """
    Orchestrates the Generation -> Validation -> Evaluation loop.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.reporter = RiskReporter(config_path)
        self.validator = OutputValidator()
        self.evaluator = ModelEvaluator(config_path)
        self.max_retries = self.config['ai_settings'].get('max_retries', 2)

        # Setup Jinja for prompt rendering (to handle feedback injection)
        self.env = Environment(loader=FileSystemLoader("templates"))
        self.report_template = self.env.get_template("report_instruction.j2")
        self.persona_template = self.env.get_template("system_persona.j2")

    def run(self, auc_stats, chi_stats, binomial_df, jeffrey_df, ttest_stats):
        """
        Runs the pipeline with self-correction.
        """
        # Prepare Context
        context = {
            "AUC_Analysis": auc_stats,
            "Chi_Square_Test": chi_stats,
            "Binomial_Test_Results": binomial_df.to_dict(orient="records"),
            "Jeffrey_Test_Results": jeffrey_df.to_dict(orient="records"),
            "Population_Stability_T_Test": ttest_stats
        }
        context_str = json.dumps(context, indent=2)
        input_hash = hashlib.sha256(context_str.encode()).hexdigest()

        current_feedback = ""
        final_output = ""
        verification_status = "Fail"
        trust_score = 0

        for attempt in range(self.max_retries + 1):
            print(f"--- Pipeline Attempt {attempt + 1}/{self.max_retries + 1} ---")

            # 1. Generate (using Template)
            # We override the ai_engine's simple prompt logic by constructing the prompt here
            # and passing it directly if ai_engine allowed, or we assume ai_engine is dumb.
            # Let's modify ai_engine later to be more flexible, or just use reporter.client here?
            # Better to reuse reporter to keep connection logic there, but we need to inject the prompt.
            # I will assume I'll update RiskReporter to accept a full 'messages' list or 'user_prompt'.
            # For now, let's render the prompt here.

            alpha = self.config['thresholds']['significance_level_alpha']
            user_prompt = self.report_template.render(
                context_data=context_str,
                alpha=alpha,
                feedback=current_feedback
            )

            system_prompt = self.persona_template.render()

            # Call AI Engine (Need to refactor AI Engine to take specific prompts)
            # For now, I'll access the client directly from reporter or update reporter.
            # Let's assume I update reporter to have a `generate_from_prompt(system, user)` method.
            # Or I can just copy the generation logic here if ai_engine is too rigid.
            # Let's stick to using reporter but we will need to refactor it.
            # I will use a protected method for now or `client` directly.

            try:
                response = self.reporter.client.chat.completions.create(
                    model=self.config['ai_settings']['model'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config['ai_settings']['temperature'],
                    max_tokens=self.config['ai_settings']['max_tokens']
                )
                generated_text = response.choices[0].message.content
            except Exception as e:
                print(f"Generation Error: {e}")
                return {
                    "report": f"Error: {str(e)}",
                    "status": "Error",
                    "score": 0,
                    "attempts": attempt + 1
                }

            # 2. Validate (Deterministic)
            passed_val, val_errors = self.validator.validate(generated_text, context_str)

            if not passed_val:
                print(f"Validation Failed: {val_errors}")
                current_feedback = f"Your previous report failed validation checks:\n" + "\n".join(val_errors)
                continue # Retry

            # 3. Evaluate (LLM Judge)
            eval_result = self.evaluator.evaluate_report(generated_text, context_str)
            trust_score = eval_result.get('score', 0)

            if not eval_result.get('pass', False):
                print(f"Evaluation Failed (Score {trust_score}): {eval_result.get('reasoning')}")
                current_feedback = f"Your previous report failed quality evaluation:\n{eval_result.get('reasoning')}"
                continue # Retry

            # Success!
            verification_status = "Pass"
            final_output = generated_text
            break

        # If loop finishes without success
        if verification_status == "Fail":
             final_output = generated_text if generated_text else "Failed to generate valid report."
             if not current_feedback: # If it was a generation error
                 current_feedback = "Unknown error."

        # 4. Audit Log
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_hash": input_hash,
            "attempts": attempt + 1,
            "final_status": verification_status,
            "trust_score": trust_score,
            "validation_errors": val_errors if not passed_val else [],
            "evaluation_reasoning": eval_result.get('reasoning') if passed_val else "Skipped",
            "final_output_snippet": final_output[:200]
        }
        logging.info(json.dumps(log_entry))

        return {
            "report": final_output,
            "status": verification_status,
            "score": trust_score,
            "attempts": attempt + 1
        }
