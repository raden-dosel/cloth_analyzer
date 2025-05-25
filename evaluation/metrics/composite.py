from torchmetrics import MetricCollection
from .bias import DemographicParityDifference
from .classification import HierarchicalF1, CategorySpecificity
from .sentiment import ContextualizedMAE, SentimentConsistency

class FashionMetricComposite(MetricCollection):
    """Combines all metrics with dynamic weighting"""
    def __init__(self, config: dict):
        metrics = {
            'hierarchical_f1': HierarchicalF1(config['hierarchy']),
            'specificity': CategorySpecificity(config['hierarchy']),
            'contextual_mae': ContextualizedMAE(config['category_weights']),
            'consistency': SentimentConsistency(),
            'fairness': DemographicParityDifference(config['demographic_categories'])
        }
        
        super().__init__(metrics)
        
    def compute(self) -> dict:
        results = super().compute()
        
        # Calculate weighted overall score
        weights = self.config.get('weights', {
            'hierarchical_f1': 0.4,
            'specificity': 0.2,
            'contextual_mae': 0.2,
            'consistency': 0.1,
            'fairness': 0.1
        })
        
        results['overall_score'] = sum(
            results[k] * weights.get(k, 0)
            for k in results if k != 'overall_score'
        )
        
        return results