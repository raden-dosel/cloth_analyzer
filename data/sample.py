import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import necessary modules

from processing.feature_engineer import FeatureEngineer

# Initialize the FeatureEngineer
engineer = FeatureEngineer()

# Sample input text
input_text = "I prefer a red cotton shirt over a blue one, but not in XL size."

# Extract features
features = engineer.extract_features(input_text)

# Log input and output
print("=== Input Text ===")
print(input_text)
print("\n=== Extracted Features ===")
print(features)