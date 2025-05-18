from pydantic import BaseModel, ValidationError, Field, validator
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class AttributeCategory(str, Enum):
    COLOR = "color"
    MATERIAL = "material"
    STYLE = "style"
    FIT = "fit"
    OCCASION = "occasion"

class DataSchema(BaseModel):
    """Defines the schema for valid preference data records"""
    text: str = Field(..., min_length=5, max_length=500)
    attributes: Dict[AttributeCategory, List[str]] = Field(
        ..., 
        description="Extracted clothing attributes by category"
    )
    sentiment: Dict[str, float] = Field(
        ..., 
        description="Sentiment scores for each attribute",
        example={"color:dark": 0.9, "material:polyester": -0.8}
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing metadata"
    )
    
    @validator('text')
    def validate_text(cls, v):
        if 'http' in v:
            raise ValueError("Text contains invalid URL")
        if len(v.split()) < 2:
            raise ValueError("Text too short")
        return v
    
    @validator('sentiment')
    def validate_sentiment_scores(cls, v):
        for key, score in v.items():
            if not (-1 <= score <= 1):
                raise ValueError(f"Invalid sentiment score {score} for {key}")
        return v

class DataValidator:
    def __init__(self, strict: bool = True):
        self.strict = strict
        self.required_attributes = {
            AttributeCategory.COLOR,
            AttributeCategory.MATERIAL
        }
    
    def validate_sample(self, sample: Dict) -> tuple:
        """Validate a single data sample against schema"""
        try:
            validated = DataSchema(**sample)
            self._validate_attribute_presence(validated.attributes)
            return True, validated.dict()
        except ValidationError as e:
            if self.strict:
                raise
            logger.warning(f"Invalid sample: {str(e)}")
            return False, str(e)
        except ValueError as e:
            logger.error(f"Data validation error: {str(e)}")
            return False, str(e)
    
    def _validate_attribute_presence(self, attributes: Dict):
        """Check for required attribute categories"""
        present = set(attributes.keys())
        missing = self.required_attributes - present
        if missing:
            raise ValueError(f"Missing required attributes: {missing}")

    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """Validate entire dataset and return statistics"""
        results = {
            'valid': [],
            'invalid': [],
            'missing_attributes': 0,
            'invalid_sentiment': 0
        }
        
        for sample in dataset:
            is_valid, result = self.validate_sample(sample)
            if is_valid:
                results['valid'].append(result)
            else:
                results['invalid'].append({
                    'sample': sample,
                    'error': result
                })
                if "Missing required attributes" in result:
                    results['missing_attributes'] += 1
                if "sentiment score" in result:
                    results['invalid_sentiment'] += 1
        
        results['validation_rate'] = len(results['valid']) / len(dataset)
        return results