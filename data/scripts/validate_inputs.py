import sys
import os
import re
from typing import List, Dict, Set

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from library.attribute_categories import ATTRIBUTE_CATEGORIES
from library.sentiment_keywords import SENTIMENT_KEYWORDS

class InputValidator:
    def __init__(self):
        # Create regex for single-word attributes
        single_word_attrs = [attr for sublist in ATTRIBUTE_CATEGORIES.values() 
                            for attr in sublist if " " not in attr]
        self.single_word_regex = re.compile(
            r"\b(" + "|".join(single_word_attrs) + r")\b", re.IGNORECASE
        )
        
        # Create regex for multi-word attributes (needs escaping)
        multi_word_attrs = [attr for sublist in ATTRIBUTE_CATEGORIES.values() 
                           for attr in sublist if " " in attr]
        if multi_word_attrs:
            self.multi_word_regex = re.compile(
                r"\b(" + "|".join(re.escape(attr) for attr in multi_word_attrs) + r")\b", 
                re.IGNORECASE
            )
        else:
            self.multi_word_regex = None
        
        # Create attribute-to-category mapping
        self.attr_to_category = {}
        for category, attrs in ATTRIBUTE_CATEGORIES.items():
            for attr in attrs:
                self.attr_to_category[attr.lower()] = category
        
        # Expanded negation phrases based on synthetic data generation
        self.negation_phrases = [
            "no", "not", "don't", "avoid", "without", "never", "n't",
            "don't want any", "please exclude", "I'm not looking for", 
            "I'd prefer not to have", "keep away from", "nothing with",
            "want to stay clear of", "rather not have", "not my style"
        ]
        
        # Compile sentiment keyword regexes
        self.positive_sentiment_regex = re.compile(
            r"\b(" + "|".join(re.escape(word) for word in SENTIMENT_KEYWORDS["positive"]) + r")\b",
            re.IGNORECASE
        )
        self.negative_sentiment_regex = re.compile(
            r"\b(" + "|".join(re.escape(word) for word in SENTIMENT_KEYWORDS["negative"]) + r")\b",
            re.IGNORECASE
        )
        
        # Common comparison phrases from synthetic data
        self.comparison_phrases = [
            "prefer", "over", "more than", "better than", "rather than", 
            "instead of", "would choose", "before", "more appealing than", 
            "suits me better than", "like more than"
        ]
    
    def find_all_attributes(self, text: str) -> Set[str]:
        """Find all attributes in text, including multi-word attributes"""
        single_word_matches = set(self.single_word_regex.findall(text.lower()))
        multi_word_matches = set()
        if self.multi_word_regex:
            multi_word_matches = set(self.multi_word_regex.findall(text.lower()))
        return single_word_matches.union(multi_word_matches)
    
    def detect_sentiments(self, text: str) -> Dict[str, List[str]]:
        """Detect positive and negative sentiment keywords in text"""
        positive = self.positive_sentiment_regex.findall(text.lower())
        negative = self.negative_sentiment_regex.findall(text.lower())
        return {
            "positive": positive,
            "negative": negative
        }
    
    def detect_comparisons(self, text: str) -> List[str]:
        """Detect comparison phrases in text"""
        return [phrase for phrase in self.comparison_phrases if phrase.lower() in text.lower()]
    
    def group_attributes_by_category(self, attributes: Set[str]) -> Dict[str, List[str]]:
        """Group attributes by their categories"""
        categorized = {}
        for attr in attributes:
            category = self.attr_to_category.get(attr.lower())
            if category:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(attr)
        return categorized
    
    def validate(self, text: str) -> Dict:
        """Validate user input against data schema"""
        errors = []
        warnings = []
        
        # Find all attributes
        mentioned_attrs = self.find_all_attributes(text)
        
        # Check for unrecognized attributes
        known_attrs = set(self.attr_to_category.keys())
        unknown_attrs = mentioned_attrs - known_attrs
        
        if unknown_attrs:
            warnings.append(f"Unrecognized attributes detected: {', '.join(unknown_attrs)}")
        
        # Detect sentiments
        sentiments = self.detect_sentiments(text)
        
        # Check for ambiguous negations
        negation_flags = [phrase for phrase in self.negation_phrases if phrase.lower() in text.lower()]
        if len(negation_flags) > 1:
            errors.append("Multiple negation terms detected - may cause interpretation conflicts")
        
        # Detect comparisons
        comparisons = self.detect_comparisons(text)
        
        # Group attributes by category
        categorized_attrs = self.group_attributes_by_category(mentioned_attrs)
        
        # Check for sentiment conflicts (mixed positive/negative without comparison)
        if sentiments["positive"] and sentiments["negative"] and not comparisons:
            warnings.append("Mixed positive and negative sentiments without clear comparison structure")
        
        # Validate positive sentiment usage
        if sentiments["positive"]:
            # Check if positive sentiment is present but no attributes are mentioned
            if not mentioned_attrs:
                warnings.append("Positive sentiment expressed but no specific attributes mentioned")
            
            # Check if positive sentiment words are potentially misaligned with attributes
            text_parts = text.lower().split()
            for pos_word in sentiments["positive"]:
                # Find position of sentiment word in text
                if pos_word in text_parts:
                    pos_index = text_parts.index(pos_word)
                    # Check if there's no attribute within 5 words of the sentiment
                    nearby_attrs = False
                    for attr in mentioned_attrs:
                        # For multi-word attributes, check the first word
                        attr_first_word = attr.lower().split()[0]
                        if attr_first_word in text_parts:
                            attr_index = text_parts.index(attr_first_word)
                            if abs(pos_index - attr_index) <= 5:
                                nearby_attrs = True
                                break
                    
                    if not nearby_attrs:
                        warnings.append(f"Positive sentiment '{pos_word}' may not be clearly associated with any attribute")
        
        # Analyze categories mentioned
        if len(categorized_attrs) > 4:
            warnings.append("Large number of attribute categories mentioned - may need narrowing down")
        
        # Calculate a sentiment score per attribute category
        sentiment_scores = {}
        if mentioned_attrs:
            # Simplified proximity analysis for sentiment-attribute association
            for category, attrs in categorized_attrs.items():
                sentiment_scores[category] = {"positive": 0, "negative": 0}
                
                for attr in attrs:
                    attr_pos = text.lower().find(attr.lower())
                    if attr_pos >= 0:
                        # Check proximity to positive/negative words
                        for pos_word in sentiments["positive"]:
                            pos_word_pos = text.lower().find(pos_word.lower())
                            if pos_word_pos >= 0 and abs(pos_word_pos - attr_pos) < 20:
                                sentiment_scores[category]["positive"] += 1
                                
                        for neg_word in sentiments["negative"]:
                            neg_word_pos = text.lower().find(neg_word.lower())
                            if neg_word_pos >= 0 and abs(neg_word_pos - attr_pos) < 20:
                                sentiment_scores[category]["negative"] += 1
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "detected_attributes": list(mentioned_attrs),
            "categorized_attributes": categorized_attrs,
            "sentiments": sentiments,
            "sentiment_scores": sentiment_scores,
            "negations": negation_flags,
            "comparisons": comparisons
        }