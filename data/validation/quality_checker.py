import numpy as np
from typing import List, Dict, Any
import logging
from collections import defaultdict
from .schema_validator import AttributeCategory, DataSchema

logger = logging.getLogger(__name__)

class DataQualityChecker:
    def __init__(self):
        self.checks = [
            self.check_missing_values,
            self.check_text_length,
            self.check_sentiment_range,
            self.check_attribute_values,
            self.check_duplicates,
            self.check_mandatory_fields
        ]
        
        # Define acceptable attribute values
        self.allowed_values = {
            AttributeCategory.COLOR: {
                'red', 'blue', 'black', 'white', 'dark', 'light'
            },
            AttributeCategory.MATERIAL: {
                'cotton', 'polyester', 'silk', 'wool', 'linen'
            }
        }

    def run_checks(self, dataset: List[Dict]) -> Dict:
        """Execute all quality checks and compile report"""
        report = defaultdict(list)
        stats = {
            'total_samples': len(dataset),
            'passed_samples': 0,
            'check_violations': defaultdict(int)
        }
        
        for sample in dataset:
            sample_passed = True
            for check in self.checks:
                passed, message = check(sample)
                if not passed:
                    report[sample['text']].append(message)
                    stats['check_violations'][check.__name__] += 1
                    sample_passed = False
            
            if sample_passed:
                stats['passed_samples'] += 1
        
        stats['pass_rate'] = stats['passed_samples'] / stats['total_samples']
        return {
            'report': dict(report),
            'statistics': stats
        }

    def check_missing_values(self, sample: Dict) -> tuple:
        """Check for missing required fields"""
        required = ['text', 'attributes', 'sentiment']
        missing = [field for field in required if field not in sample]
        if missing:
            return False, f"Missing fields: {missing}"
        return True, ""

    def check_text_length(self, sample: Dict) -> tuple:
        """Validate text length constraints"""
        text = sample['text']
        if len(text) < 10 or len(text) > 500:
            return False, f"Invalid text length: {len(text)}"
        return True, ""

    def check_sentiment_range(self, sample: Dict) -> tuple:
        """Verify sentiment scores are within [-1, 1]"""
        invalid = []
        for attr, score in sample['sentiment'].items():
            if not (-1 <= score <= 1):
                invalid.append(f"{attr}: {score}")
        if invalid:
            return False, f"Invalid sentiment scores: {', '.join(invalid)}"
        return True, ""

    def check_attribute_values(self, sample: Dict) -> tuple:
        """Validate attribute values against allowed list"""
        invalid = defaultdict(list)
        for category, values in sample['attributes'].items():
            allowed = self.allowed_values.get(category, set())
            if not allowed:
                continue  # Skip categories without restrictions
            for value in values:
                if value.lower() not in allowed:
                    invalid[category].append(value)
        
        if invalid:
            messages = [f"{k}: {v}" for k, v in invalid.items()]
            return False, f"Invalid attribute values - {', '.join(messages)}"
        return True, ""

    def check_duplicates(self, sample: Dict) -> tuple:
        """Check for duplicate attribute values"""
        duplicates = defaultdict(list)
        for category, values in sample['attributes'].items():
            unique = set()
            for value in values:
                if value in unique:
                    duplicates[category].append(value)
                unique.add(value)
        
        if duplicates:
            messages = [f"{k}: {v}" for k, v in duplicates.items()]
            return False, f"Duplicate values - {', '.join(messages)}"
        return True, ""

    def check_mandatory_fields(self, sample: Dict) -> tuple:
        """Ensure required attributes are present"""
        mandatory = {AttributeCategory.COLOR, AttributeCategory.MATERIAL}
        present = set(sample['attributes'].keys())
        missing = mandatory - present
        if missing:
            return False, f"Missing mandatory attributes: {missing}"
        return True, ""

    def generate_quality_report(self, dataset: List[Dict]) -> str:
        """Generate human-readable quality report"""
        results = self.run_checks(dataset)
        report = [
            "Data Quality Report",
            "===================",
            f"Total Samples: {results['statistics']['total_samples']}",
            f"Pass Rate: {results['statistics']['pass_rate']:.1%}",
            "\nTop Issues:"
        ]
        
        # Sort issues by frequency
        issues = sorted(
            results['statistics']['check_violations'].items(),
            key=lambda x: x[1], 
            reverse=True
        )
        
        for check_name, count in issues:
            report.append(f"- {check_name}: {count} violations")
        
        return '\n'.join(report)