import os
import sys

# Adjust parent directory in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from transformers import DistilBertTokenizerFast
from typing import List, Dict
import numpy as np
import logging
from .cleaning import TextCleaner

logger = logging.getLogger(__name__)

class ClothingTokenizer:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.cleaner = TextCleaner()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.special_tokens = {
            '[URL]', '[USER]', '[NEGATION]', '[MATERIAL]', '[COLOR]'
        }
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add domain-specific special tokens"""
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(self.special_tokens)
        })

    def tokenize_with_cleaning(self, text: str) -> Dict:
        """Full processing pipeline"""
        cleaned_text = self.cleaner.clean_text(text)
        negations = self.cleaner.detect_negations(cleaned_text)
        
        # Tokenize with special markers
        marked_text = self._insert_special_markers(cleaned_text, negations)
        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_offsets_mapping=True
        )
        
        # Create label alignment
        labels = self._create_labels(encoding, negations)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'offset_mapping': encoding['offset_mapping'],
            'labels': labels
        }

    def _insert_special_markers(self, text: str, negations: List[Dict]) -> str:
        """Insert special tokens for important patterns"""
        # Mark negations
        for neg in reversed(negations):
            text = f"{text[:neg['start']]}[NEGATION] {text[neg['start']:neg['end']]} [END_NEG]{text[neg['end']:]}"
        
        # Add other domain markers (implement similar patterns for materials/colors)
        return text

    def _create_labels(self, encoding: Dict, negations: List[Dict]) -> List[int]:
        """Create token-level labels for NER-style task"""
        labels = np.zeros(len(encoding['input_ids']), dtype=int)
        char_labels = self._create_char_level_labels(encoding, negations)
        
        # Convert character-level labels to token-level
        for ix, (start, end) in enumerate(encoding['offset_mapping']):
            if start == end == 0:
                continue  # Special tokens
            label_slice = char_labels[start:end]
            if sum(label_slice) > 0:
                labels[ix] = max(set(label_slice), key=list(label_slice).count)
        return labels.tolist()

    def _create_char_level_labels(self, encoding: Dict, negations: List[Dict]) -> np.array:
        """Create character-level labeling array"""
        text = self.tokenizer.decode(encoding['input_ids'], skip_special_tokens=True)
        char_labels = np.zeros(len(text), dtype=int)
        
        # Label negation spans
        for neg in negations:
            start = neg['start']
            end = neg['end']
            char_labels[start:end] = 1  # 1 = negation context
            
        # Add other labeling logic for materials, colors etc
        return char_labels

    def decode_with_labels(self, input_ids: List[int], labels: List[int]) -> str:
        """Decode tokens with label annotations"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        decoded = []
        for token, label in zip(tokens, labels):
            if label != 0:
                decoded.append(f"[{label}]{token}[/{label}]")
            else:
                decoded.append(token)
        return ' '.join(decoded).replace(' ##', '')