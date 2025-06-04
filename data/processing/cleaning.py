import re
import html
from functools import lru_cache
from typing import List, Dict
import logging
import unicodedata

logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self):
        self._compile_patterns()
        self.contraction_map = self._load_contractions()
        
    def _compile_patterns(self):
        """Precompile regex patterns for efficiency"""
        self.whitespace_re = re.compile(r'\s+')
        self.url_re = re.compile(r'https?://\S+|www\.\S+')
        self.mention_re = re.compile(r'@\w+')
        self.hashtag_re = re.compile(r'#(\w+)')
        self.non_printable_re = re.compile(
            f'[^{re.escape("".join(chr(i) for i in range(32, 127)))}]'
        )
        self.negation_re = re.compile(
            r"\b(?:not|no|don't|doesn't|isn't|aren't|wasn't|weren't|can't|couldn't)\b[\w\s]+", 
            re.IGNORECASE
        )

    def _load_contractions(self) -> Dict[str, str]:
        """Load common English contractions"""
        return {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "i'm": "i am",
            "it's": "it is",
            "they're": "they are"
            # Add more contractions as needed
        }

    @lru_cache(maxsize=10000)
    def clean_text(self, text: str) -> str:
        """Main text cleaning pipeline"""
        try:
            text = self._basic_clean(text)
            text = self._handle_special_patterns(text)
            text = self._expand_contractions(text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning text: {text[:50]}... - {str(e)}")
            return ""

    def _basic_clean(self, text: str) -> str:
        """Initial normalization and cleanup"""
        text = html.unescape(text)
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        text = self.url_re.sub('[URL]', text)
        text = self.mention_re.sub('[USER]', text)
        text = self.hashtag_re.sub(r'\1', text)
        text = self.non_printable_re.sub(' ', text)
        text = re.sub(r'!', '', text)  # Remove exclamation marks
        return text

    def _handle_special_patterns(self, text: str) -> str:
        """Handle clothing-specific patterns"""
        text = re.sub(r'\b(\d+) ?%', r'\1 percent', text)  # 100% -> 100 percent
        text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)    # 100ml -> 100 ml
        text = re.sub(r'\bs/m\b', 'small medium', text)
        text = re.sub(r'\b(x{2,})\b', lambda m: f"{len(m.group(1))} x", text)
        return text

    def _expand_contractions(self, text: str) -> str:
        """Replace contractions with expanded forms"""
        for cont, expanded in self.contraction_map.items():
            text = re.sub(r'\b' + cont + r'\b', expanded, text)
        return text

    def detect_negations(self, text: str) -> List[Dict]:
        """Identify and categorize negation phrases"""
        negations = []
        for match in self.negation_re.finditer(text):
            span = match.span()
            negations.append({
                'text': match.group(),
                'start': span[0],
                'end': span[1],
                'type': 'negation'
            })
        return negations