import spacy
import jsonlines
from collections import defaultdict
from pathlib import Path

class NegationErrorLogger:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.errors = []
        self.error_counts = defaultdict(int)
        
    def _detect_negation_scope(self, text: str) -> List[dict]:
        """Use dependency parsing to find negation scope"""
        doc = self.nlp(text)
        negations = []
        
        for token in doc:
            if token.dep_ == "neg":
                # Get the negation head and children
                scope = [token.head] + list(token.head.children)
                negations.append({
                    "negation_word": token.text,
                    "scope": [t.text for t in scope],
                    "start_char": token.idx,
                    "end_char": token.head.idx + len(token.head.text)
                })
                
        return negations
    
    def log_errors(self, text: str, pred_attributes: dict, true_attributes: dict):
        """Detect and log negation handling errors"""
        negations = self._detect_negation_scope(text)
        
        for negation in negations:
            scope_text = " ".join(negation["scope"]).lower()
            error_found = False
            
            # Check if any negated attributes were incorrectly predicted
            for category in ["positive_preferences", "negative_preferences"]:
                for attr in pred_attributes.get(category, []):
                    if attr in scope_text and category != self._expected_category(attr, scope_text):
                        error_found = True
                        self.error_counts[(category, attr)] += 1
                        
            if error_found:
                self.errors.append({
                    "text": text,
                    "negation_scope": scope_text,
                    "predicted_attributes": pred_attributes,
                    "true_attributes": true_attributes,
                    "negation_details": negation
                })

    def _expected_category(self, attr: str, scope: str) -> str:
        """Determine if attribute should be positive or negative"""
        return "negative_preferences" if attr in scope else "positive_preferences"

    def save_errors(self, output_dir: str = "error_analysis/"):
        """Save errors to JSONL file and generate summary"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save raw errors
        with jsonlines.open(f"{output_dir}/negation_errors.jsonl", "w") as writer:
            writer.write_all(self.errors)
            
        # Generate summary report
        summary = []
        for (category, attr), count in self.error_counts.items():
            summary.append({
                "category": category,
                "attribute": attr,
                "error_count": count,
                "error_rate": count / len(self.errors) if self.errors else 0
            })
            
        pd.DataFrame(summary).to_csv(
            f"{output_dir}/negation_error_summary.csv", index=False
        )
        
        return self.errors