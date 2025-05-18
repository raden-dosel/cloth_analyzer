import random
import nlpaug.augmenter.word as naw
from typing import List, Dict
from datasets import Dataset
import logging
from data.processing.cleaning import TextCleaner

logger = logging.getLogger(__name__)

class DataAugmenter:
    def __init__(self, aug_p: float = 0.3):
        self.aug_p = aug_p
        self.cleaner = TextCleaner()
        
        # Initialize augmentation methods
        self.synonym_aug = naw.SynonymAug(aug_p=aug_p)
        self.contextual_aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', 
            action="substitute",
            aug_p=aug_p
        )
        
        self.augmenters = [
            self._synonym_replacement,
            self._contextual_replacement,
            self._attribute_swap,
            self._negation_addition
        ]

    def augment_dataset(self, dataset: Dataset, num_augments: int = 2) -> Dataset:
        """Augment dataset with multiple strategies"""
        augmented_samples = []
        
        for sample in dataset:
            augmented_samples.append(sample)  # Keep original
            for _ in range(num_augments):
                aug_sample = self._augment_sample(sample.copy())
                if aug_sample:
                    augmented_samples.append(aug_sample)

        return Dataset.from_list(augmented_samples)

    def _augment_sample(self, sample: Dict) -> Dict:
        """Apply augmentation pipeline to a single sample"""
        try:
            # Augment text
            original_text = sample["text"]
            aug_text = self._augment_text(original_text)
            
            if aug_text == original_text:
                return None  # Skip no-change augmentations
                
            # Update attributes if needed
            sample["text"] = aug_text
            sample["metadata"]["augmented"] = True
            sample["metadata"]["original_text"] = original_text
            
            return sample
        except Exception as e:
            logger.error(f"Error augmenting sample: {str(e)}")
            return None

    def _augment_text(self, text: str) -> str:
        """Apply text augmentation strategies"""
        # Randomly select augmentation method
        aug_method = random.choice(self.augmenters)
        return aug_method(text)

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms"""
        return self.synonym_aug.augment(text)

    def _contextual_replacement(self, text: str) -> str:
        """Context-aware word replacement using BERT"""
        return self.contextual_aug.augment(text)

    def _attribute_swap(self, text: str) -> str:
        """Swap clothing attributes while preserving meaning"""
        cleaned = self.cleaner.clean_text(text)
        attributes = self.cleaner.detect_attributes(cleaned)
        
        if not attributes:
            return text
            
        # Replace a random attribute
        attr_to_replace = random.choice(attributes)
        category = attr_to_replace["category"]
        new_value = random.choice(self._get_alternatives(category))
        
        return text.replace(
            attr_to_replace["text"], 
            new_value, 
            1  # Only replace first occurrence
        )

    def _negation_addition(self, text: str) -> str:
        """Add or remove negation patterns"""
        if "not" in text.lower() or "no" in text.lower():
            # Remove negation
            return self._remove_negation(text)
        else:
            # Add negation to random attribute
            return self._add_negation(text)

    def _add_negation(self, text: str) -> str:
        """Add negation to a random attribute phrase"""
        phrases = self.cleaner.detect_attributes(self.cleaner.clean_text(text))
        if not phrases:
            return text
            
        target = random.choice(phrases)
        return text.replace(
            target["text"], 
            f"not {target['text']}", 
            1
        )

    def _remove_negation(self, text: str) -> str:
        """Remove negation from text"""
        return text.replace("not ", "").replace("no ", "").replace("n't ", " ")

    def _get_alternatives(self, category: str) -> List[str]:
        """Get alternative values for an attribute category"""
        # This should match your attribute pool from generator.py
        alternatives = {
            "color": ["red", "blue", "black", "white", "navy", "beige"],
            "material": ["cotton", "polyester", "silk", "wool", "linen"],
            "style": ["formal", "casual", "sporty", "vintage"],
            "fit": ["slim", "loose", "relaxed", "tailored"],
            "occasion": ["wedding", "office", "party", "outdoor"]
        }
        return alternatives.get(category.lower(), [])