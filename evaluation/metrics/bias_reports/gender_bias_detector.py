import spacy
import pandas as pd
from collections import defaultdict
from pathlib import Path

class GenderBiasAnalyzer:
    def __init__(self, 
                 gender_terms_path: str = "configs/gender_terms.yaml",
                 output_dir: str = "bias_reports/"):
        self.nlp = spacy.load("en_core_web_lg")
        self.gender_terms = self._load_gender_terms(gender_terms_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bias_cases = []
        self.stats = defaultdict(lambda: defaultdict(int))

    def _load_gender_terms(self, path: str) -> dict:
        """Load gender-related terms from config"""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def _detect_gendered_context(self, text: str) -> dict:
        """Analyze text for gendered language and entities"""
        doc = self.nlp(text)
        detected = {
            "explicit_terms": [],
            "named_entities": [],
            "coref_clusters": []
        }

        # Detect explicit gender terms
        for token in doc:
            lower_token = token.text.lower()
            if lower_token in self.gender_terms['explicit']:
                detected["explicit_terms"].append(lower_token)

        # Detect gendered named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "NORP"]:
                if any(gender in ent.text.lower() for gender in ['he', 'she', 'mr', 'mrs']):
                    detected["named_entities"].append((ent.text, ent.label_))

        return detected

    def analyze_recommendations(self, 
                               user_text: str, 
                               recommendations: list, 
                               ideal_distribution: dict):
        """Analyze gender bias in recommendations"""
        gender_context = self._detect_gendered_context(user_text)
        recommendation_genders = self._classify_recommendations(recommendations)

        # Calculate bias metrics
        gender_counts = pd.Series(recommendation_genders).value_counts(normalize=True)
        bias_scores = {}
        for gender in ['male', 'female', 'neutral']:
            ideal = ideal_distribution.get(gender, 0)
            actual = gender_counts.get(gender, 0)
            bias_scores[gender] = actual - ideal

        # Log case
        self.bias_cases.append({
            "user_text": user_text,
            "detected_gender_terms": gender_context,
            "recommendation_distribution": dict(gender_counts),
            "bias_scores": bias_scores
        })

        # Update statistics
        for gender, score in bias_scores.items():
            self.stats[gender]['total'] += 1
            if abs(score) > 0.1:  # Threshold for significant bias
                self.stats[gender]['biased_cases'] += 1

    def _classify_recommendations(self, recommendations: list) -> list:
        """Classify recommended items by gender association"""
        genders = []
        for item in recommendations:
            description = item['description'].lower()
            gender = 'neutral'
            
            # Check for explicit gender markers
            if any(term in description for term in self.gender_terms['male']):
                gender = 'male'
            elif any(term in description for term in self.gender_terms['female']):
                gender = 'female'
                
            genders.append(gender)
        return genders

    def generate_report(self):
        """Generate comprehensive gender bias report"""
        # Save raw cases
        pd.DataFrame(self.bias_cases).to_csv(
            self.output_dir / "gender_bias_cases.csv", index=False
        )

        # Calculate summary statistics
        summary = {
            gender: {
                'total_cases': stats['total'],
                'biased_cases': stats['biased_cases'],
                'bias_rate': stats['biased_cases'] / stats['total'] if stats['total'] else 0
            }
            for gender, stats in self.stats.items()
        }

        # Generate markdown report
        with open(self.output_dir / "gender_bias_analysis.md", "w") as f:
            f.write("# Gender Bias Analysis Report\n\n")
            f.write("## Overall Statistics\n")
            f.write(pd.DataFrame(summary).to_markdown() + "\n\n")
            f.write("## Example Biased Cases\n")
            for case in self.bias_cases[:5]:
                f.write(f"### Input: {case['user_text']}\n")
                f.write(f"- **Detected Terms**: {case['detected_gender_terms']}\n")
                f.write(f"- **Recommendation Distribution**: {case['recommendation_distribution']}\n")
                f.write(f"- **Bias Scores**: {case['bias_scores']}\n\n")
        
        return summary