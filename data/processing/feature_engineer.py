import os
import sys

# Adjust parent directory in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from typing import List, Dict, Tuple
import re
import spacy
from spacy.matcher import PhraseMatcher
import logging
from data.library.attribute_categories import ATTRIBUTE_CATEGORIES

from .cleaning import TextCleaner

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.nlp = spacy.load("en_core_web_sm")
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Create matchers for clothing attributes"""
        self.color_properties = ATTRIBUTE_CATEGORIES.color_properties
        self.color_orientation = ATTRIBUTE_CATEGORIES.color_orientation
        self.material_properties = ATTRIBUTE_CATEGORIES.material_properties
        self.pattern_properties = ATTRIBUTE_CATEGORIES.material_properties
        self.occasion = ATTRIBUTE_CATEGORIES.color_properties
        self.style = ATTRIBUTE_CATEGORIES.color_orientation
        self.weather_suitability = ATTRIBUTE_CATEGORIES.material_properties
        self.fit = ATTRIBUTE_CATEGORIES.material_properties
        self.embellishments = ATTRIBUTE_CATEGORIES.material_properties
        self.sleeve_length = ATTRIBUTE_CATEGORIES.material_properties
        self.neckline = ATTRIBUTE_CATEGORIES.color_properties
        self.pants_length = ATTRIBUTE_CATEGORIES.color_orientation
        self.skirt_length = ATTRIBUTE_CATEGORIES.material_properties
        self.dress_length = ATTRIBUTE_CATEGORIES.material_properties
        self.shirt_type = ATTRIBUTE_CATEGORIES.material_properties
        self.dress_type = ATTRIBUTE_CATEGORIES.material_properties
        self.pants_type = ATTRIBUTE_CATEGORIES.material_properties
        self.jacket_type = ATTRIBUTE_CATEGORIES.material_properties
        
        
        
        # Create phrase matchers
        self.color_properties_matcher = PhraseMatcher(self.nlp.vocab)
        self.color_orientation_matcher = PhraseMatcher(self.nlp.vocab)
        self.material_properties_matcher = PhraseMatcher(self.nlp.vocab)
        self.pattern_properties_matcher = PhraseMatcher(self.nlp.vocab)
        self.occasion_matcher = PhraseMatcher(self.nlp.vocab)
        self.style_matcher = PhraseMatcher(self.nlp.vocab)
        self.weather_suitability_matcher = PhraseMatcher(self.nlp.vocab)
        self.fit_matcher = PhraseMatcher(self.nlp.vocab)
        self.embellishments_matcher = PhraseMatcher(self.nlp.vocab)
        self.sleeve_length_matcher = PhraseMatcher(self.nlp.vocab)
        self.neckline_matcher = PhraseMatcher(self.nlp.vocab)
        self.pants_length_matcher = PhraseMatcher(self.nlp.vocab)
        self.skirt_length_matcher = PhraseMatcher(self.nlp.vocab)
        self.dress_length_matcher = PhraseMatcher(self.nlp.vocab)
        self.shirt_type_matcher = PhraseMatcher(self.nlp.vocab)
        self.dress_type_matcher = PhraseMatcher(self.nlp.vocab)
        self.pants_type_matcher = PhraseMatcher(self.nlp.vocab)
        self.jacket_type_matcher = PhraseMatcher(self.nlp.vocab)

        for term_list, matcher in [
            (self.color_properties, self.color_properties_matcher),
            (self.color_orientation, self.color_orientation_matcher),
            (self.material_properties, self.material_properties_matcher),
            (self.pattern_properties, self.pattern_properties_matcher),
            (self.occasion, self.occasion_matcher),
            (self.style, self.style_matcher),
            (self.weather_suitability, self.weather_suitability_matcher),
            (self.fit, self.fit_matcher),
            (self.embellishments, self.embellishments_matcher),
            (self.sleeve_length, self.sleeve_length_matcher),
            (self.neckline, self.neckline_matcher),
            (self.pants_length, self.pants_length_matcher),
            (self.skirt_length, self.skirt_length_matcher),
            (self.dress_length, self.dress_length_matcher),
            (self.shirt_type, self.shirt_type_matcher),
            (self.dress_type, self.dress_type_matcher),
            (self.pants_type, self.pants_type_matcher),
            (self.jacket_type, self.jacket_type_matcher),
        ]:
            patterns = [self.nlp.make_doc(text) for text in term_list]
            matcher.add("Attributes", None, *patterns)

    def extract_features(self, text: str) -> Dict:
        """Main feature extraction pipeline"""
        cleaned_text = self.cleaner.clean_text(text)
        doc = self.nlp(cleaned_text)
        
        features = {
            'color_properties': [],
            'color_orientation': [],
            'material_properties': [],
            'pattern_properties': [],
            'occasion': [],
            'style': [],
            'weather_suitability': [],
            'fit': [],
            'embellishments': [],
            'neckline': [],
            'sleeve_length': [],
            'pants_length': [],
            'skirt_length': [],
            'dress_length': [],
            'shirt_type': [],
            'jacket_type': [],
            'dress_type': [],
            'pants_type': [],
            'positive': [],
            'negative': [],
            'comparison': [],
            'negation': [],
        }
        
        # Extract using multiple methods
        self._match_phrases(doc, features)
        self._parse_negations(doc, features)
        self._extract_comparisons(cleaned_text, features)
        
        return features

    def _match_phrases(self, doc, features: Dict):
        """Extract using spaCy matchers"""
        matches = {
            'COLOR': self.color_matcher,
            'MATERIAL': self.material_matcher,
            'STYLE': self.style_matcher
        }
        
        for label, matcher in matches.items():
            for _, start, end in matcher(doc):
                span = doc[start:end]
                features[label.lower() + 's'].append({
                    'text': span.text,
                    'start': span.start_char,
                    'end': span.end_char
                })

    def _parse_negations(self, doc, features: Dict):
        """Identify negated attributes using dependency parsing"""
        for token in doc:
            if token.dep_ == 'neg' and token.head.pos_ == 'VERB':
                # Find negated objects
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'attr'):
                        features['negations'].append({
                            'attribute': child.text,
                            'negator': token.text,
                            'context': child.sent.text
                        })

    def _extract_measures(self, doc) -> List[Dict]:
        """Extract measurements using regex patterns"""
        measures = []
        patterns = {
            'size': r'\b(x?s|m|l|x{1,3}l)\b',
            'dimension': r'\b(\d+[\s-]?(?:inch|cm|centimeter|"))\b',
            'weight': r'\b(\d+[\s-]?(?:kg|pound|lb))\b'
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, doc.text):
                measures.append({
                    'type': label,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        return measures

    def _extract_comparisons(self, text: str, features: Dict):
        """Identify preference comparisons"""
        comparisons = re.finditer(
            r'\b(prefer|rather|instead of|over)\b', 
            text, 
            re.IGNORECASE
        )
        
        for match in comparisons:
            features.setdefault('comparisons', []).append({
                'text': match.group(),
                'context': text[max(0, match.start()-50):match.end()+50]
            })