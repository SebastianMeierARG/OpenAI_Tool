import re
import json

class OutputValidator:
    """
    Responsible for deterministic safety and structure checks.
    """
    def __init__(self):
        # Mock regex for Account Numbers (e.g., 8-12 digits)
        self.pii_regex = re.compile(r'\b\d{8,12}\b')
        self.required_headers = [
            "Executive Summary",
            "AUC Analysis",
            "Calibration Analysis",
            "Population Stability",
            "Conclusion"
        ]

    def validate(self, text, context_data):
        """
        Runs all validation checks.
        Returns: (passed: bool, errors: list)
        """
        errors = []

        # 1. Safety Filter (PII)
        if self.check_safety(text):
            errors.append("Safety Violation: Potential PII (account numbers) detected.")

        # 2. Structure Validation
        missing_headers = self.check_structure(text)
        if missing_headers:
            errors.append(f"Structure Violation: Missing required headers: {', '.join(missing_headers)}")

        # 3. Data Consistency (Basic Number Check)
        # extracting numbers from context
        consistency_error = self.check_consistency(text, context_data)
        if consistency_error:
            errors.append(f"Data Consistency Violation: {consistency_error}")

        return len(errors) == 0, errors

    def check_safety(self, text):
        """
        Returns True if PII detected.
        """
        # We assume any sequence of 8-12 digits is a potential account number violation
        # unless it looks like a date or something specific we whitelist.
        # For this exercise, strict regex.
        # But wait, context data might have numbers?
        # The requirement says "Ensure no PII (e.g., mocked regex for account numbers) leaked into the final output."
        # If the input doesn't have PII, the output shouldn't hallucinate it.
        # If input has it, it should probably be masked before LLM.
        # Assuming input is clean stats.
        matches = self.pii_regex.findall(text)
        return len(matches) > 0

    def check_structure(self, text):
        """
        Returns list of missing headers.
        """
        missing = []
        for header in self.required_headers:
            if header not in text:
                missing.append(header)
        return missing

    def check_consistency(self, text, context_data):
        """
        Checks if specific key numbers from context appear in the text.
        This is a heuristic check.
        """
        # Load context to get key metrics
        try:
            data = json.loads(context_data)
        except json.JSONDecodeError:
            return "Context data is not valid JSON."

        # Let's check AUC difference specifically as an example of a critical number
        # logic: "If the data says 'AUC is 0.85' but the text says 'AUC is 0.90', this validation must FAIL."

        # We need to find the numbers in the text and see if they match.
        # Since text is unstructured, it's hard to know which number refers to what.
        # Strategy: Ensure that the EXACT numbers from the critical stats appear in the text.

        # 1. Check AUC Diff
        if "AUC_Analysis" in data and "Latest_AUC_Diff" in data["AUC_Analysis"]:
            val = data["AUC_Analysis"]["Latest_AUC_Diff"]
            # Formatting issues: 0.02 vs 0.0200. We search for rounded string.
            val_str = f"{val:.4f}"
            if val_str not in text and f"{val:.2f}" not in text:
                 return f"AUC Difference ({val_str}) not found in text."

        # 2. Check Chi-Square P-Value
        if "Chi_Square_Test" in data and "P_Value" in data["Chi_Square_Test"]:
             val = data["Chi_Square_Test"]["P_Value"]
             val_str = f"{val:.4f}"
             if val_str not in text and f"{val:.2f}" not in text:
                 # It's possible the LLM says "P-Value < 0.0001" if it's very small.
                 # This simple check might be too strict, but per requirements:
                 # "compare them against the calculated statistics".
                 # We will return error if exact match not found.
                 return f"Chi-Square P-Value ({val_str}) not found in text."

        return None
