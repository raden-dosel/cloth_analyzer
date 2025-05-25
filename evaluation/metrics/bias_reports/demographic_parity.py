import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path

class DemographicParityCalculator:
    def __init__(self, 
                 demographic_categories: list,
                 output_dir: str = "bias_reports/"):
        self.demographic_categories = demographic_categories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def calculate_parity(self, y_true: pd.Series, y_pred: pd.Series, 
                        demographic_data: pd.DataFrame):
        """Calculate demographic parity metrics"""
        parity_report = {}
        
        for category in self.demographic_categories:
            unique_groups = demographic_data[category].unique()
            if len(unique_groups) < 2:
                continue
                
            group1, group2 = unique_groups[:2]
            mask1 = demographic_data[category] == group1
            mask2 = demographic_data[category] == group2
            
            parity_score = self._statistical_parity_difference(
                y_pred[mask1], y_pred[mask2]
            )
            
            parity_report[category] = {
                "group1": group1,
                "group2": group2,
                "parity_score": parity_score,
                "group1_mean": y_pred[mask1].mean(),
                "group2_mean": y_pred[mask2].mean()
            }
            
            self.results.append({
                "demographic_category": category,
                **parity_report[category]
            })

        # Save results
        pd.DataFrame(self.results).to_csv(
            self.output_dir / "demographic_parity.csv", index=False
        )
        
        return parity_report

    def _statistical_parity_difference(self, preds1: np.array, preds2: np.array) -> float:
        """Calculate SPD between two groups"""
        return preds1.mean() - preds2.mean()

    def generate_fairness_report(self, threshold: float = 0.1):
        """Generate fairness compliance report"""
        df = pd.DataFrame(self.results)
        df['bias_detected'] = abs(df['parity_score']) > threshold
        
        with open(self.output_dir / "fairness_report.md", "w") as f:
            f.write("# Demographic Fairness Report\n\n")
            f.write("## Parity Scores\n")
            f.write(df.to_markdown(index=False) + "\n\n")
            f.write(f"## Bias Detection (Threshold: {threshold})\n")
            f.write(df.groupby('demographic_category')['bias_detected'].value_counts().to_markdown())
        
        return df