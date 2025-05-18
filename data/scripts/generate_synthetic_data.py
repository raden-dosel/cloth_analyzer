import sys
import os
import random
from faker import Faker
from datasets import Dataset
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from library.attribute_categories import ATTRIBUTE_CATEGORIES
from library.sentiment_keywords import SENTIMENT_KEYWORDS



fake = Faker()




# Expanded sentiment keywords for more diversity

def generate_sample():
    """Generate one synthetic training sample with annotations"""
    # Randomly select 2-4 attributes to include
    attributes = random.sample(list(ATTRIBUTE_CATEGORIES.items()), k=random.randint(2,4))
    
    sentence_parts = []
    annotations = []
    
    # Build sentence with mixed sentiments
    for category, values in attributes:
        value = random.choice(values)
        sentiment = random.choice(["positive", "negative"])
        
        # Construct phrase with sentiment
        if sentiment == "positive":
            template = random.choice([
                f"I really {random.choice(SENTIMENT_KEYWORDS['positive'])} {value} {category.split('_')[0]}",
                f"Looking for {value} ones",
                f"Must have {value} features",
                f"I'm searching for something {value}",
                f"Do you have anything in {value}?",
                f"I'm interested in {value} options",
                f"{value} would be perfect for me",
                f"I'm drawn to {value} {category.split('_')[0]}",
                f"Could you recommend something {value}?",
                f"I tend to go for {value} items",
                f"I'm partial to {value} {category.split('_')[0]}",
                f"Ideally, it would be {value}",
                f"{value} is exactly what I need"
            ])
        else:
            template = random.choice([
                f"I {random.choice(SENTIMENT_KEYWORDS['negative'])} {value}",
                f"Avoid {value} at all costs",
                f"Not interested in {value}",
                f"Please don't show me anything {value}",
                f"{value} doesn't work for me",
                f"I'm not a fan of {value} {category.split('_')[0]}",
                f"I'd rather not have {value}",
                f"{value} is not my style",
                f"I've had bad experiences with {value}",
                f"I'm looking for alternatives to {value}",
                f"Anything but {value}, please",
                f"{value} makes me uncomfortable",
                f"I'm trying to stay away from {value}"
            ])
        
        sentence_parts.append(template)
        annotations.append({
            "phrase": value,
            "category": category,
            "sentiment": sentiment
        })
    
    # Add 20% chance of negation with expanded variations
    if random.random() < 0.2:
        negation = random.choice([
            "no ", "don't want any ", "avoid ", "please exclude ", 
            "I'm not looking for ", "I'd prefer not to have ", 
            "without any ", "keep away from ", "nothing with ",
            "I want to stay clear of "
        ])
        sentence_parts.insert(0, negation)
    
    # Add 30% chance of comparison with expanded variations
    if random.random() < 0.3:
        compare_attr = random.choice(list(ATTRIBUTE_CATEGORIES.values()))
        comparison = random.choice([
            f"prefer {compare_attr[0]} over {compare_attr[1]}",
            f"like {compare_attr[0]} more than {compare_attr[1]}",
            f"{compare_attr[0]} is better than {compare_attr[1]}",
            f"{compare_attr[0]} rather than {compare_attr[1]}",
            f"{compare_attr[0]} instead of {compare_attr[1]}",
            f"would choose {compare_attr[0]} before {compare_attr[1]}",
            f"find {compare_attr[0]} more appealing than {compare_attr[1]}",
            f"{compare_attr[0]} suits me better than {compare_attr[1]}"
        ])
        sentence_parts.append(comparison)
        annotations.extend([
            {"phrase": compare_attr[0], "category": category, "sentiment": "positive"},
            {"phrase": compare_attr[1], "category": category, "sentiment": "negative"}
        ])
    
    return {
        "text": " ".join(sentence_parts),
        "annotations": annotations
    }

# Generate dataset
dataset = Dataset.from_pandas(pd.DataFrame([generate_sample() for _ in range(1000000)]))
dataset.save_to_disk("data/synthetic")