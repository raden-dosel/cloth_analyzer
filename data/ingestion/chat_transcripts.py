import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Union
from jsonschema import ValidationError
import pandas as pd
from .base_loader import BaseDataLoader, DataRecord
from ..validation.schema_validator import DataValidator

# Configure logging
logger = logging.getLogger(__name__)

class ChatTranscriptLoader(BaseDataLoader):
    SUPPORTED_FORMATS = ['.csv', '.json', '.parquet']
    
    def __init__(self, data_path: Union[str, Path]):
        super().__init__(data_path)
        self.validator = DataValidator()
        self._detect_format()

    def _detect_format(self) -> None:
        """Auto-detect file format from extension"""
        self.file_format = self.data_path.suffix.lower()
        if self.file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {self.file_format}. Supported: {self.SUPPORTED_FORMATS}")

    def load_data(self) -> List[DataRecord]:
        """Load chat data from various formats"""
        loader = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.parquet': self._load_parquet
        }.get(self.file_format)
        
        if not loader:
            raise NotImplementedError(f"Loader for {self.file_format} not implemented")
            
        raw_data = loader()
        return self._parse_records(raw_data)

    def _load_csv(self) -> List[dict]:
        """Load CSV with flexible delimiter detection"""
        try:
            return pd.read_csv(self.data_path, delimiter=None, engine='python').to_dict('records')
        except Exception as e:
            logger.error(f"CSV loading failed: {str(e)}")
            raise

    def _load_json(self) -> List[dict]:
        """Load JSON data with schema validation"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of records")
            
        return data

    def _load_parquet(self) -> List[dict]:
        """Load Parquet efficiently"""
        return pd.read_parquet(self.data_path).to_dict('records')

    def _parse_records(self, raw_data: List[dict]) -> List[DataRecord]:
        """Convert raw data to validated DataRecords"""
        records = []
        for idx, item in enumerate(raw_data):
            try:
                record = DataRecord(
                    text=item['message'],
                    source=item.get('source', 'unknown'),
                    metadata={
                        'timestamp': item.get('timestamp'),
                        'user_id': item.get('user_id')
                    }
                )
                self.validator.validate_sample(record.__dict__)
                records.append(record)
            except KeyError as e:
                logger.warning(f"Skipping record {idx}: Missing key {str(e)}")
            except ValidationError as e:
                logger.warning(f"Skipping invalid record {idx}: {str(e)}")
        return records

    def get_stats(self) -> Dict[str, float]:
        """Calculate basic dataset statistics"""
        data = self.load_data()
        return {
            'total_records': len(data),
            'avg_text_length': sum(len(r.text) for r in data) / len(data),
            'sources': {r.source for r in data}
        }

    def sample_data(self, n: int = 5) -> List[DataRecord]:
        """Get first n records"""
        return self.load_data()[:n]

    # Additional chat-specific methods
    def filter_by_source(self, source: str) -> List[DataRecord]:
        """Filter records by data source"""
        return [r for r in self.load_data() if r.source == source]

    def extract_conversations(self) -> Dict[str, List[DataRecord]]:
        """Group messages by conversation ID"""
        conversations = {}
        for record in self.load_data():
            conv_id = record.metadata.get('conversation_id')
            if conv_id:
                conversations.setdefault(conv_id, []).append(record)
        return conversations