# Full pipeline usage example
import sys
import os
from datasets import load_from_disk
from data.preprocessing.preprocessing import dataset_to_dataloader
from data.scripts.validate_inputs import InputValidator


parent_dir = os.path.dirname(os.path.abspath(__file__))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 1. Generate data (only needed once)
# generate_synthetic_data.main()

# 2. Load and preprocess
dataset = load_from_disk("data/synthetic_dataset")
dataloader = dataset_to_dataloader(dataset)

# 3. Validate real input
validator = InputValidator()
user_input = "I want a cotton shirt but no polyester    " 
validation_result = validator.validate(user_input)

print(f"Validation: {validation_result}")