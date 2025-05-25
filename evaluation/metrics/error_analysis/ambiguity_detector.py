import spacy
import pandas as pd
from collections import defaultdict
from typing import List, Dict
from pathlib import Path

class AmbiguityDetector:
    def __init__(self, ambiguous_terms_path: str = "configs/ambiguous_terms.yaml"):
        self.nlp = spacy.load("en_core_web_lg")
        self.ambiguous_terms = self._load_ambiguous_terms(ambiguous_terms_path)
        self.ambiguous_cases = []
        
    def _load_ambiguous_terms(self, path: str) -> Dict[str, List[str]]:
        """Load ambiguous terms with their possible categories"""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
        
    def _get_context_vectors(self, text: str, window_size: int = 3) -> Dict[str, float]:
        """Get semantic context vectors for ambiguity resolution"""
        doc = self.nlp(text)
        context_vectors = {}
        
        for token in doc:
            if token.text.lower() in self.ambiguous_terms:
                context = [
                    t.text.lower()
                    for t in doc[max(0, token.i-window_size):token.i+window_size]
                    if not t.is_stop and t.is_alpha
                ]
                context_vectors[token.text] = self.nlp(" ".join(context)).vector
                
        return context_vectors

    def detect(self, text: str, pred_attributes: dict, true_attributes: dict):
        """Detect and analyze ambiguous phrases"""
        doc = self.nlp(text.lower())
        detected = defaultdict(list)
        
        # Check for ambiguous terms
        for term, categories in self.ambiguous_terms.items():
            if term in text.lower():
                context_vec = self._get_context_vectors(text).get(term, None)
                
                # Check model's interpretation
                model_interpretation = [
                    cat for cat in categories
                    if any(a in pred_attributes.get(cat, []) for a in categories[cat])
                ]
                
                # Check ground truth interpretation
                true_interpretation = [
                    cat for cat in categories
                    if any(a in true_attributes.get(cat, []) for a in categories[cat])
                ]
                
                confidence = self._calculate_confidence(context_vec, true_interpretation)
                
                detected[term].append({
                    "text": text,
                    "term": term,
                    "possible_categories": list(categories.keys()),
                    "model_interpretation": model_interpretation,
                    "true_interpretation": true_interpretation,
                    "confidence": confidence,
                    "needs_review": model_interpretation != true_interpretation
                })
        
        if detected:
            self.ambiguous_cases.extend(detected.values())

    def _calculate_confidence(self, context_vec, true_category):
        """Calculate confidence based on semantic similarity"""
        if not context_vec or not true_category:
            return 0.0
            
        # Get average vector for true category terms
        category_vec = sum(
            self.nlp(c).vector for c in self.ambiguous_terms[true_category]
        ) / len(self.ambiguous_terms[true_category])
        
        return float(np.dot(context_vec, category_vec) / 
                   (np.linalg.norm(context_vec) * np.linalg.norm(category_vec)))

    def generate_report(self, output_dir: str = "error_analysis/"):
        """Generate ambiguity analysis report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        df = pd.DataFrame([
            {**case, "term": term} 
            for term_cases in self.ambiguous_cases 
            for term, case in term_cases.items()
        ])
        
        # Save detailed report
        df.to_csv(f"{output_dir}/ambiguity_cases.csv", index=False)
        
        # Generate summary markdown
        with open(f"{output_dir}/ambiguity_report.md", "w") as f:
            f.write("# Ambiguity Analysis Report\n\n")
            f.write(f"**Total Ambiguous Cases**: {len(df)}\n\n")
            f.write("## Most Common Ambiguities\n")
            f.write(df.groupby("term").size().to_markdown() + "\n\n")
            f.write("## Interpretation Accuracy\n")
            f.write(df.groupby("term")["needs_review"].mean().to_markdown())
            
        return df