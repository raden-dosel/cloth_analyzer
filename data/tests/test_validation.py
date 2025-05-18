import pytest
from data.validation.schema_validator import DataValidator, DataSchema
from data.validation.quality_checker import DataQualityChecker

@pytest.fixture
def valid_sample():
    return {
        "text": "I prefer cotton shirts in dark colors for formal occasions",
        "attributes": {
            "color": ["dark"],
            "material": ["cotton"],
            "occasion": ["formal"]
        },
        "sentiment": {
            "color:dark": 0.8,
            "material:cotton": 0.9,
            "occasion:formal": 0.7
        }
    }

@pytest.fixture
def invalid_sample():
    return {
        "text": "I like it",
        "attributes": {"color": []},
        "sentiment": {"color:red": 1.5}
    }

class TestDataValidator:
    def test_valid_sample(self, valid_sample):
        validator = DataValidator()
        is_valid, _ = validator.validate_sample(valid_sample)
        assert is_valid

    def test_invalid_sample(self, invalid_sample):
        validator = DataValidator(strict=False)
        is_valid, errors = validator.validate_sample(invalid_sample)
        assert not is_valid
        assert "text too short" in errors
        assert "sentiment score" in errors

    def test_missing_attributes(self):
        validator = DataValidator()
        sample = {
            "text": "This is a test",
            "attributes": {"style": ["casual"]},
            "sentiment": {"style:casual": 0.5}
        }
        is_valid, errors = validator.validate_sample(sample)
        assert not is_valid
        assert "Missing required attributes" in errors

class TestDataQualityChecker:
    @pytest.fixture
    def checker(self):
        return DataQualityChecker()

    def test_quality_checks(self, checker, valid_sample, invalid_sample):
        report = checker.run_checks([valid_sample, invalid_sample])
        assert report['statistics']['total_samples'] == 2
        assert report['statistics']['passed_samples'] == 1
        assert 'check_text_length' in report['statistics']['check_violations']

    def test_attribute_validation(self, checker):
        sample = {
            "text": "I want neon pink polka dot pants",
            "attributes": {"color": ["neon pink"], "material": ["polka dot"]},
            "sentiment": {"color:neon pink": 0.9, "material:polka dot": 0.8}
        }
        report = checker.run_checks([sample])
        assert 'color: [\'neon pink\']' in str(report['report'])

    def test_sentiment_range(self, checker):
        sample = {
            "text": "Valid text with invalid sentiment",
            "attributes": {"color": ["red"]},
            "sentiment": {"color:red": 1.1}
        }
        report = checker.run_checks([sample])
        assert 'invalid sentiment scores' in str(report['report'])