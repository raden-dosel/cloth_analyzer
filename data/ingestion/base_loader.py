from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, List, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataRecord:
    text: str
    source: str
    metadata: Dict[str, Any] = None

class BaseDataLoader(ABC):
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.validate_data_source()

    def validate_data_source(self) -> None:
        """Check if data source exists and is accessible"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
        logger.info(f"Validated data source at {self.data_path}")

    @abstractmethod
    def load_data(self) -> List[DataRecord]:
        """Load and parse raw data into structured records"""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        """Calculate basic dataset statistics"""
        raise NotImplementedError

    @abstractmethod
    def sample_data(self, n: int = 5) -> List[DataRecord]:
        """Get sample records for inspection"""
        raise NotImplementedError