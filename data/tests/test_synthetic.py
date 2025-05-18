import pytest
from data.synthetic.generator import SyntheticDataGenerator
from data.synthetic.augmenter import DataAugmenter

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
        assert len(augmented_attrs['color']) >= len(original_attrs['color'])
        assert len(augmented_attrs['material']) >= len(original_attrs['material'])

    def test_negation_handling(self, augmenter):
        text = "I want cotton shirts"
        augmented = augmenter._negation_addition(text)
        assert "not" in augmented or "without" in augmented
        
        neg_text = "I don't like polyester"
        cleaned = augmenter._remove_negation(neg_text)
        assert "don't" not in cleaned