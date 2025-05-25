import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict

class CulturalCoverageAnalyzer:
    def __init__(self, 
                 cultural_taxonomy_path: str = "configs/cultural_taxonomy.yaml",
                 output_dir: str = "bias_reports/"):
        self.taxonomy = self._load_taxonomy(cultural_taxonomy_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coverage_stats = defaultdict(lambda: defaultdict(int))
        self.misclassified = []

    def _load_taxonomy(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def analyze_text(self, text: str, pred_attributes: dict, true_attributes: dict):
        """Analyze cultural coverage in model predictions"""
        detected_cultures = self._detect_cultural_terms(text)
        
        for culture, terms in detected_cultures.items():
            self.coverage_stats[culture]['total'] += 1
            
            # Check if model recognized cultural context
            recognized = any(
                any(t in pred_attributes.get('style', []) or 
                    t in pred_attributes.get('pattern', []))
                for t in terms['expected']
            )
            
            if recognized:
                self.coverage_stats[culture]['recognized'] += 1
            else:
                self.misclassified.append({
                    "text": text,
                    "culture": culture,
                    "expected_terms": terms['expected'],
                    "predicted_terms": pred_attributes
                })

    def _detect_cultural_terms(self, text: str) -> dict:
        """Detect cultural terms in input text"""
        detected = {}
        text_lower = text.lower()
        
        for culture, data in self.taxonomy.items():
            matched_terms = [
                term for term in data['terms']
                if term in text_lower
            ]
            if matched_terms:
                detected[culture] = {
                    "matched": matched_terms,
                    "expected": data['expected_attributes']
                }
        
        return detected

    def generate_report(self):
        """Generate cultural coverage report"""
        # Calculate coverage rates
        coverage_df = pd.DataFrame([
            {
                "culture": culture,
                "total_occurrences": stats['total'],
                "recognition_rate": stats.get('recognized', 0) / stats['total'] if stats['total'] else 0,
                "common_terms": ', '.join(self.taxonomy[culture]['terms'][:3])
            }
            for culture, stats in self.coverage_stats.items()
        ])

        # Save reports
        coverage_df.to_csv(self.output_dir / "cultural_coverage.csv", index=False)
        pd.DataFrame(self.misclassified).to_csv(
            self.output_dir / "cultural_misclassifications.csv", index=False
        )

        # Generate markdown report
        with open(self.output_dir / "cultural_coverage_report.md", "w") as f:
            f.write("# Cultural Coverage Analysis\n\n")
            f.write("## Recognition Rates by Culture\n")
            f.write(coverage_df.to_markdown(index=False) + "\n\n")
            f.write("## Common Misclassifications\n")
            for example in self.misclassified[:5]:
                f.write(f"**Input**: {example['text']}\n\n")
                f.write(f"- **Culture**: {example['culture']}\n")
                f.write(f"- **Expected**: {example['expected_terms']}\n")
                f.write(f"- **Predicted**: {example['predicted_terms']}\n\n")
        
        return coverage_df