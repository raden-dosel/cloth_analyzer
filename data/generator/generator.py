import random
from faker import Faker
from typing import Dict, List
from datasets import Dataset
import logging
from enum import Enum
from data.validation.schema_validator import AttributeCategory, SentimentLabel

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    COMPARISON = "comparison"
    NEGATION = "negation"

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.faker = Faker()
        self.faker.seed_instance(seed)
        random.seed(seed)
        
        self.attribute_pool = {
            AttributeCategory.COLOR: ["red", "blue", "black", "white", "navy", "beige"],
            AttributeCategory.MATERIAL: ["cotton", "polyester", "silk", "wool", "linen"],
            AttributeCategory.STYLE: ["formal", "casual", "sporty", "vintage"],
            AttributeCategory.FIT: ["slim", "loose", "relaxed", "tailored"],
            AttributeCategory.OCCASION: ["wedding", "office", "party", "outdoor"]
        }
        
        self.templates = {
            PreferenceType.POSITIVE: [
                "I really like {attribute} {item}",
                "Looking for {item} in {attribute}",
                "Prefer {attribute} for {occasion} events"
            ],
            PreferenceType.NEGATIVE: [
                "I don't like {attribute} {item}",
                "Avoid {attribute} materials",
                "Not a fan of {attribute} colors"
            ],
            PreferenceType.COMPARISON: [
                "Prefer {attribute1} over {attribute2}",
                "{attribute1} is better than {attribute2}",
                "Choose {attribute1} instead of {attribute2}"
            ],
            PreferenceType.NEGATION: [
                "No {attribute} please",
                "Without any {attribute}",
                "I never want {attribute} in my clothes"
            ]
        }

    def _select_attribute(self, category: AttributeCategory) -> str:
        return random.choice(self.attribute_pool[category])

    def _generate_item(self) -> str:
        return random.choice(["shirt", "dress", "pants", "jacket", "skirt"])

    def _generate_occasion(self) -> str:
        return random.choice(self.attribute_pool[AttributeCategory.OCCASION])

    def _generate_base_sample(self) -> Dict:
        """Generate core attributes for a sample"""
        return {
            "text": "",
            "attributes": {cat.value: [] for cat in AttributeCategory},
            "sentiment": {},
            "metadata": {
                "source": "synthetic",
                "generation_type": None
            }
        }

    def _add_attribute(self, sample: Dict, category: AttributeCategory, value: str, sentiment: float):
        """Helper to add attributes with sentiment"""
        sample["attributes"][category.value].append(value)
        key = f"{category.value}:{value}"
        sample["sentiment"][key] = sentiment

    def generate_sample(self) -> Dict:
        """Generate a single synthetic sample"""
        sample = self._generate_base_sample()
        pref_type = random.choices(
            list(PreferenceType),
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]

        try:
            if pref_type == PreferenceType.POSITIVE:
                self._handle_positive(sample)
            elif pref_type == PreferenceType.NEGATIVE:
                self._handle_negative(sample)
            elif pref_type == PreferenceType.COMPARISON:
                self._handle_comparison(sample)
            else:
                self._handle_negation(sample)

            sample["metadata"]["generation_type"] = pref_type.value
            return sample
        except Exception as e:
            logger.error(f"Error generating sample: {str(e)}")
            return None

    def _handle_positive(self, sample: Dict):
        """Generate positive preference sample"""
        category = random.choice([
            AttributeCategory.COLOR,
            AttributeCategory.MATERIAL,
            AttributeCategory.STYLE
        ])
        attribute = self._select_attribute(category)
        item = self._generate_item()
        
        template = random.choice(self.templates[PreferenceType.POSITIVE])
        sample["text"] = template.format(
            attribute=attribute,
            item=item,
            occasion=self._generate_occasion()
        )
        self._add_attribute(sample, category, attribute, 0.9 + random.random()/10)

    def _handle_negative(self, sample: Dict):
        """Generate negative preference sample"""
        category = random.choice([
            AttributeCategory.MATERIAL,
            AttributeCategory.FIT,
            AttributeCategory.COLOR
        ])
        attribute = self._select_attribute(category)
        
        template = random.choice(self.templates[PreferenceType.NEGATIVE])
        sample["text"] = template.format(attribute=attribute)
        self._add_attribute(sample, category, attribute, -0.8 - random.random()/10)

    def _handle_comparison(self, sample: Dict):
        """Generate comparison preference sample"""
        category = random.choice([
            AttributeCategory.MATERIAL,
            AttributeCategory.COLOR,
            AttributeCategory.STYLE
        ])
        attrs = random.sample(self.attribute_pool[category], 2)
        
        template = random.choice(self.templates[PreferenceType.COMPARISON])
        sample["text"] = template.format(attribute1=attrs[0], attribute2=attrs[1])
        self._add_attribute(sample, category, attrs[0], 0.7 + random.random()/10)
        self._add_attribute(sample, category, attrs[1], -0.6 - random.random()/10)

    def _handle_negation(self, sample: Dict):
        """Generate negation pattern sample"""
        category = random.choice([
            AttributeCategory.MATERIAL,
            AttributeCategory.FIT
        ])
        attribute = self._select_attribute(category)
        
        template = random.choice(self.templates[PreferenceType.NEGATION])
        sample["text"] = template.format(attribute=attribute)
        self._add_attribute(sample, category, attribute, -1.0)

    def generate_dataset(self, num_samples: int = 1000) -> Dataset:
        """Generate full synthetic dataset"""
        samples = []
        while len(samples) < num_samples:
            sample = self.generate_sample()
            if sample:
                samples.append(sample)
                if len(samples) % 100 == 0:
                    logger.info(f"Generated {len(samples)}/{num_samples} samples")
        return Dataset.from_list(samples)