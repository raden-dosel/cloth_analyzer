import os
import sys

# Adjust parent directory in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pytest
from data.generator.generator import SyntheticDataGenerator
from data.generator.augmenter import DataAugmenter

@pytest.fixture
def generator():
    return SyntheticDataGenerator(seed=42)

@pytest.fixture
def augmenter():
    return DataAugmenter(aug_p=0.3)

class TestSyntheticGeneration:
    def test_sample_generation(self, generator):
        sample = generator.generate_sample()
        assert len(sample['text']) > 10
        assert 'color' in sample['attributes']
        assert len(sample['sentiment']) > 0

    def test_dataset_creation(self, generator):
        dataset = generator.generate_dataset(10)
        assert len(dataset) == 10
        assert all('metadata' in sample for sample in dataset)

    def test_sentiment_ranges(self, generator):
        for _ in range(100):
            sample = generator.generate_sample()
            for score in sample['sentiment'].values():
                assert -1 <= score <= 1

class TestDataAugmentation:
    def test_augmentation(self, generator, augmenter):
        original = generator.generate_sample()
        augmented = augmenter._augment_sample(original.copy())
        assert augmented is not None
        assert augmented['text'] != original['text']
        assert augmented['metadata']['augmented']

    def test_attribute_preservation(self, generator, augmenter):
        original = generator.generate_sample()
        augmented = augmenter._augment_sample(original.copy())
        original_attrs = original['attributes']
        augmented_attrs = augmented['attributes']
        
        # Verify core attributes remain
        assert len(augmented_attrs['color_properties']) >= len(original_attrs['color_properties'])
        assert len(augmented_attrs['color_orientation']) >= len(original_attrs['color_orientation'])
        assert len(augmented_attrs['material_properties']) >= len(original_attrs['material_properties'])
        assert len(augmented_attrs['pattern_properties']) >= len(original_attrs['pattern_properties'])
        assert len(augmented_attrs['occasion']) >= len(original_attrs['occasion'])
        assert len(augmented_attrs['style']) >= len(original_attrs['style'])
        assert len(augmented_attrs['weather_suitability']) >= len(original_attrs['color_properties'])
        assert len(augmented_attrs['fit']) >= len(original_attrs['color_orientation'])
        assert len(augmented_attrs['embellishments']) >= len(original_attrs['material_properties'])
        assert len(augmented_attrs['neckline']) >= len(original_attrs['pattern_properties'])
        assert len(augmented_attrs['sleeve_length']) >= len(original_attrs['occasion'])
        assert len(augmented_attrs['pants_length']) >= len(original_attrs['style'])
        assert len(augmented_attrs['skirt_length']) >= len(original_attrs['color_properties'])
        assert len(augmented_attrs['dress_length']) >= len(original_attrs['color_orientation'])
        assert len(augmented_attrs['shirt_type']) >= len(original_attrs['material_properties'])
        assert len(augmented_attrs['jacket_type']) >= len(original_attrs['pattern_properties'])
        assert len(augmented_attrs['dress_type']) >= len(original_attrs['occasion'])
        assert len(augmented_attrs['pants_type']) >= len(original_attrs['style'])

    def test_negation_handling(self, augmenter):
        text = "I want cotton shirts"
        augmented = augmenter._negation_addition(text)
        assert "not" in augmented or "without" in augmented
        
        neg_text = "I don't like polyester"
        cleaned = augmenter._remove_negation(neg_text)
        assert "don't" not in cleaned