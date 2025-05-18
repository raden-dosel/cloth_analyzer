import sys
import os
import matplotlib.pyplot as plt


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from library.attribute_categories import ATTRIBUTE_CATEGORIES


def plot_class_distribution(dataset):
    counts = {k:0 for k in ATTRIBUTE_CATEGORIES}
    for sample in dataset:
        for ann in sample["annotations"]:
            counts[ann["category"]] += 1
    plt.bar(counts.keys(), counts.values())  