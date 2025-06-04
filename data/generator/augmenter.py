import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import necessary modules
import random
import nlpaug.augmenter.word as naw
from typing import List, Dict
from datasets import Dataset
import logging


from data.processing.cleaning import TextCleaner
from data.library.attribute_categories import ATTRIBUTE_CATEGORIES

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
        negation_patterns = [
            # Basic negations
            "not", "no", "n't", "never",
            # Verb-based negations
            "don't", "doesn't", "didn't", "won't", "wouldn't",
            "cannot", "can't", "couldn't",
            "shouldn't", "mustn't", "isn't", "aren't",
            # Action-based negations
            "avoid", "exclude", "reject", "refuse",
            "dislike", "hate", "despise",
            # Preference-based negations
            "uninterested", "opposed", "against",
            "rather not", "prefer not",
            # State-based negations
            "without", "lacking", "missing",
            # Attitude-based negations
            "uncomfortable with", "dissatisfied with",
            "unhappy with", "disappointed with",
            # Additional negation patterns
            "am not a fan of", "don't enjoy", "can't stand",
            "would rather avoid", "prefer not to have",
            "am not interested in", "would prefer not to wear",
            "doesn't match my style", "doesn't suit my taste",
            "isn't my type of", "doesn't work with my wardrobe",
            "rarely choose", "wouldn't pick", "try not to select",
            "typically avoid", "generally skip", "usually pass on",
            "am uncomfortable wearing", "don't feel confident in",
            "am not comfortable with", "doesn't align with my personal style",
            "doesn't reflect my fashion sense", "isn't part of my preferred aesthetic",
            "doesn't represent my taste in clothing"
        ]
        
        # Check if any negation pattern exists
        if any(pattern in text.lower() for pattern in negation_patterns):
            return self._remove_negation(text)
        else:
            return self._add_negation(text)

    def _add_negation(self, text: str) -> str:
        """Add negation to a random attribute phrase"""
        phrases = self.cleaner.detect_attributes(self.cleaner.clean_text(text))
        if not phrases:
            return text
        
        target = random.choice(phrases)
        negation_templates = [
            # Direct negations
            "not {}",
            "don't like {}",
            "don't want {}",
            "definitely not {}",
            "absolutely not {}",
            
            # Preference-based negations
            "would rather avoid {}",
            "prefer not to have {}",
            "am not interested in {}",
            "would prefer not to wear {}",
            "am not looking for {}",
            "don't typically go for {}",
            
            # Emotional negations
            "dislike {}",
            "hate {}",
            "can't stand {}",
            "really don't enjoy {}",
            "am not a fan of {}",
            
            # Action-based negations
            "tend to avoid {}",
            "stay away from {}",
            "steer clear of {}",
            "try to avoid {}",
            "generally skip {}",
            "usually pass on {}",
            
            # State-based negations
            "am uncomfortable with {}",
            "am not fond of {}",
            "am not keen on {}",
            "feel awkward in {}",
            "don't feel confident in {}",
            "am not comfortable wearing {}",
            
            # Style preference negations
            "doesn't match my style {}",
            "doesn't suit my taste {}",
            "isn't my type of {}",
            "doesn't work with my wardrobe {}",
            
            # Complex negations
            "would prefer something other than {}",
            "am looking for alternatives to {}",
            "would rather wear something besides {}",
            "am trying to move away from {}",
            "would appreciate options different from {}",
            "am hoping to find something unlike {}",
            
            # Conditional negations
            "rarely choose {} if I can help it",
            "wouldn't pick {} given the choice",
            "try not to select {} when possible",
            "typically avoid {} when shopping",
            
            # Personal style negations
            "doesn't align with my personal style {}",
            "doesn't reflect my fashion sense {}",
            "isn't part of my preferred aesthetic {}",
            "doesn't represent my taste in clothing {}"
        ]
        
        negation_pattern = random.choice(negation_templates)
        return text.replace(
            target["text"],
            negation_pattern.format(target["text"]),
            1
        )

    def _remove_negation(self, text: str) -> str:
        """Remove negation from text"""
        patterns_to_remove = [
            # Basic negations
            ("not ", ""),
            ("no ", ""),
            ("n't ", " "),
            ("never ", "always "),
            # Verb-based negations
            ("don't ", "do "),
            ("doesn't ", "does "),
            ("didn't ", "did "),
            ("won't ", "will "),
            ("wouldn't ", "would "),
            ("cannot ", "can "),
            ("can't ", "can "),
            ("couldn't ", "could "),
            # Action-based negations
            ("avoid ", "prefer "),
            ("exclude ", "include "),
            ("reject ", "accept "),
            ("refuse ", "choose "),
            # Preference-based negations
            ("dislike ", "like "),
            ("hate ", "love "),
            ("despise ", "appreciate "),
            # State-based negations
            ("uninterested ", "interested "),
            ("opposed to ", "in favor of "),
            ("against ", "for "),
            ("rather not ", "rather "),
            ("prefer not ", "prefer "),
            # Complex negations
            ("uncomfortable with ", "comfortable with "),
            ("dissatisfied with ", "satisfied with "),
            ("unhappy with ", "happy with "),
            ("disappointed with ", "pleased with "),
            ("without ", "with "),
            ("lacking ", "having "),
            ("missing ", "including "),
            # Additional negation patterns
            ("am not a fan of ", "am a fan of "),
            ("don't enjoy ", "enjoy "),
            ("can't stand ", "can appreciate "),
            ("would rather avoid ", "would rather embrace "),
            ("prefer not to have ", "prefer to have "),
            ("am not interested in ", "am interested in "),
            ("would prefer not to wear ", "would prefer to wear "),
            ("doesn't match my style ", "matches my style "),
            ("doesn't suit my taste ", "suits my taste "),
            ("isn't my type of ", "is my type of "),
            ("doesn't work with my wardrobe ", "works with my wardrobe "),
            ("rarely choose ", "often choose "),
            ("wouldn't pick ", "would pick "),
            ("try not to select ", "try to select "),
            ("typically avoid ", "typically choose "),
            ("generally skip ", "generally include "),
            ("usually pass on ", "usually go for "),
            ("am uncomfortable wearing ", "am comfortable wearing "),
            ("don't feel confident in ", "feel confident in "),
            ("am not comfortable with ", "am comfortable with "),
            ("doesn't align with my personal style ", "aligns with my personal style "),
            ("doesn't reflect my fashion sense ", "reflects my fashion sense "),
            ("isn't part of my preferred aesthetic ", "is part of my preferred aesthetic "),
            ("doesn't represent my taste in clothing ", "represents my taste in clothing ")
        ]
        
        result = text
        for pattern, replacement in patterns_to_remove:
            result = result.replace(pattern, replacement)
        return result

    def _get_alternatives(self, category: str) -> List[str]:
        """Get alternative values for an attribute category"""
        # This should match your attribute pool from generator.py
        alternatives = {
            "color_properties": ATTRIBUTE_CATEGORIES.color_properties,
            "color_orientation": ATTRIBUTE_CATEGORIES.color_orientation,
            "material_properties": ATTRIBUTE_CATEGORIES.material_properties,
            "pattern_properties": ATTRIBUTE_CATEGORIES.pattern_properties,
            "occasion": ATTRIBUTE_CATEGORIES.occasion,
            "style": ATTRIBUTE_CATEGORIES.style,
            "weather_suitability": ATTRIBUTE_CATEGORIES.weather_suitability,
            "fit": ATTRIBUTE_CATEGORIES.fit,
            "embellishments": ATTRIBUTE_CATEGORIES.embellishments,
            "neckline": ATTRIBUTE_CATEGORIES.neckline,
            "sleeve_length": ATTRIBUTE_CATEGORIES.sleeve_length,
            "pants_length": ATTRIBUTE_CATEGORIES.pants_length,
            "skirt_length": ATTRIBUTE_CATEGORIES.skirt_length,
            "dress_length": ATTRIBUTE_CATEGORIES.dress_length,
            "shirt_type": ATTRIBUTE_CATEGORIES.shirt_type,
            "jacket_type": ATTRIBUTE_CATEGORIES.jacket_type,
            "dress_type": ATTRIBUTE_CATEGORIES.dress_type,
            "pants_type": ATTRIBUTE_CATEGORIES.pants_type,
        }
        return alternatives.get(category.lower(), [])