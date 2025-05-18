import pytest
from data.processing.cleaning import TextCleaner
from data.processing.tokenization import ClothingTokenizer
from data.processing.feature_engineer import FeatureEngineer

@pytest.fixture
def cleaner():
    return TextCleaner()

@pytest.fixture
def tokenizer():
    return ClothingTokenizer()
 
@pytest.fixture
def engineer():
    return FeatureEngineer()

class TestTextCleaning:
    def test_basic_cleaning(self, cleaner):
        dirty_text = "Check out our SALE! Visit https://example.com"
        cleaned = cleaner.clean_text(dirty_text)
        assert "[URL]" in cleaned
        assert "sale" in cleaned.lower()
        assert "!" not in cleaned

    def test_contraction_expansion(self, cleaner):
        text = "I don't like it's fabric"
        cleaned = cleaner.clean_text(text)
        assert "do not" in cleaned
        assert "it is" in cleaned

    def test_negation_detection(self, cleaner):
        text = "Not too tight and no polyester please"
        negations = cleaner.detect_negations(text)
        assert len(negations) == 2
        assert "not too tight" in [n['text'] for n in negations]

class TestTokenization:
    def test_tokenization(self, tokenizer):
        text = "No polyester, prefer cotton"
        result = tokenizer.tokenize_with_cleaning(text)
        assert len(result['input_ids']) == 128
        assert 1 in result['labels']  # Negation label present

    def test_special_markers(self, tokenizer):
        text = "I want [MATERIAL] cotton [COLOR] blue shirts"
        result = tokenizer.tokenize_with_cleaning(text)
        decoded = tokenizer.tokenizer.decode(result['input_ids'])
        assert "[MATERIAL]" in decoded
        assert "[COLOR]" in decoded

class TestFeatureEngineering:
    def test_attribute_extraction(self, engineer):
        text = "Formal navy blue wool suit"
        features = engineer.extract_features(text)
        assert "navy" in features['colors']
        assert "wool" in features['materials']
        assert "formal" in features['styles']

    def test_negation_parsing(self, engineer):
        text = "I don't want silk material"
        features = engineer.extract_features(text)
        assert len(features['negations']) > 0
        assert features['negations'][0]['attribute'] == "silk"

    def test_measurement_extraction(self, engineer):
        text = "Size 42 pants with 32-inch legs"
        features = engineer.extract_features(text)
        assert any(m['type'] == 'size' for m in features['measures'])
        assert any('inch' in m['value'] for m in features['measures'])