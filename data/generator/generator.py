import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import necessary modules
import random
from faker import Faker
from typing import Dict, List
from datasets import Dataset
import logging
from enum import Enum
from data.validation.schema_validator import AttributeCategory, SentimentLabel
from data.library.attribute_categories import ATTRIBUTE_CATEGORIES

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    COMPARISON = "comparison"
    NEGATION = "negation"

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.faker = Faker()
        self.faker.seed_instance(seed)
        random.seed(seed)
        
        self.attribute_pool = {
            AttributeCategory.COLOR: ATTRIBUTE_CATEGORIES.color_properties,
            AttributeCategory.COLOR_ORIENTATION: ATTRIBUTE_CATEGORIES.color_orientation,
            AttributeCategory.MATERIAL: ATTRIBUTE_CATEGORIES.material_properties,
            AttributeCategory.PATTERN_PROPERTIES: ATTRIBUTE_CATEGORIES.pattern_properties,
            AttributeCategory.OCCASION: ATTRIBUTE_CATEGORIES.occasion,
            AttributeCategory.STYLE: ATTRIBUTE_CATEGORIES.style,
            AttributeCategory.WEATHER_SUITABILITY: ATTRIBUTE_CATEGORIES.weather_suitability,
            AttributeCategory.FIT: ATTRIBUTE_CATEGORIES.fit,
            AttributeCategory.EMBELLISHMENTS: ATTRIBUTE_CATEGORIES.embellishments,
            AttributeCategory.NECKLINE: ATTRIBUTE_CATEGORIES.neckline,
            AttributeCategory.SLEEVE_LENGTH: ATTRIBUTE_CATEGORIES.sleeve_length,
            AttributeCategory.PANTS_LENGTH: ATTRIBUTE_CATEGORIES.pants_length,
            AttributeCategory.SKIRT_LENGTH: ATTRIBUTE_CATEGORIES.skirt_length,
            AttributeCategory.DRESS_LENGTH: ATTRIBUTE_CATEGORIES.dress_length,
            AttributeCategory.SHIRT_TYPE: ATTRIBUTE_CATEGORIES.shirt_type,
            AttributeCategory.JACKET_TYPE: ATTRIBUTE_CATEGORIES.jacket_type,
            AttributeCategory.DRESS_TYPE: ATTRIBUTE_CATEGORIES.dress_type,
            AttributeCategory.PANTS_TYPE: ATTRIBUTE_CATEGORIES.pants_type,
            
            }
        
        self.templates = {
            PreferenceType.POSITIVE: [
            "I really like {attribute} {item}",
            "Looking for {item} in {attribute}",
            "Prefer {attribute} for {occasion} events",
            "I would love a {item} in {attribute}",
            "Must have {attribute} features",
            "I'm searching for something {attribute}",
            "Do you have anything in {attribute}?",
            "I'm interested in {attribute} options",
            "Show me {attribute} styles",
            "I like {attribute} {item}s",
            "I tend to go for {attribute} items",
            "I'm partial to {attribute} {item}s",
            "Ideally, it would be {attribute}",
            "{attribute} is exactly what I need",
            "I love {attribute} {item}s",
            "I enjoy {attribute} styles",
            "I appreciate {attribute} materials",
            "I prefer {attribute} colors",
            "I like {attribute} patterns",
            "I enjoy {attribute} fits",
            "I love {attribute} embellishments",
            "I prefer {attribute} necklines",
            "I like {attribute} sleeve lengths",
            "I enjoy {attribute} pants lengths",
            "I love {attribute} skirt lengths",
            "I prefer {attribute} dress lengths",
            "I like {attribute} shirt types",
            "I enjoy {attribute} jacket types",
            "I love {attribute} dress types",
            "I prefer {attribute} pants types",
            "I like {attribute} styles for {occasion}",
            "I enjoy {attribute} materials for {occasion}",
            "I love {attribute} colors for {occasion}",
            "I prefer {attribute} patterns for {occasion}",
            "I like {attribute} fits for {occasion}",
            "I enjoy {attribute} embellishments for {occasion}",
            "I appreciate {attribute} details for {occasion}",
            "I adore {attribute} designs",
            "I'm fond of {attribute} {item}s",
            "I gravitate towards {attribute} styles",
            "I cherish {attribute} materials",
            "I admire {attribute} patterns",
            "I favor {attribute} fits",
            "I treasure {attribute} embellishments",
            "I value {attribute} necklines",
            "I fancy {attribute} sleeve lengths",
            "I delight in {attribute} pants lengths",
            "I relish {attribute} skirt lengths",
            "I prize {attribute} dress lengths",
            "I am drawn to {attribute} shirt types",
            "I am keen on {attribute} jacket types",
            "I am captivated by {attribute} dress types",
            "I am enthusiastic about {attribute} pants types",
            "I am passionate about {attribute} styles for {occasion}",
            "I am enchanted by {attribute} materials for {occasion}",
            "I am charmed by {attribute} colors for {occasion}",
            "I am thrilled by {attribute} patterns for {occasion}",
            "I am impressed by {attribute} fits for {occasion}",
            "I am fascinated by {attribute} embellishments for {occasion}",
            "I am intrigued by {attribute} details for {occasion}",
            "I crave {attribute} designs",
            "I admire the elegance of {attribute} styles",
            "I find {attribute} patterns appealing",
            "I am drawn to the uniqueness of {attribute} materials",
            "I appreciate the versatility of {attribute} fits",
            "I am inspired by {attribute} embellishments",
            "I love the sophistication of {attribute} necklines",
            "I enjoy the comfort of {attribute} sleeve lengths",
            "I prefer the practicality of {attribute} pants lengths",
            "I adore the charm of {attribute} skirt lengths",
            "I am captivated by the beauty of {attribute} dress lengths",
            "I am impressed by the quality of {attribute} shirt types",
            "I am fond of the style of {attribute} jacket types",
            "I am enthusiastic about the design of {attribute} dress types",
            "I am passionate about the fit of {attribute} pants types",
            "I am enchanted by the creativity of {attribute} styles for {occasion}",
            "I am charmed by the innovation of {attribute} materials for {occasion}",
            "I am thrilled by the vibrancy of {attribute} colors for {occasion}",
            "I am impressed by the intricacy of {attribute} patterns for {occasion}",
            "I am fascinated by the functionality of {attribute} fits for {occasion}",
            "I am intrigued by the artistry of {attribute} embellishments for {occasion}",
            "I am captivated by the elegance of {attribute} details for {occasion}",
            "I am drawn to the timelessness of {attribute} designs",
            "I value the uniqueness of {attribute} styles",
            "I cherish the creativity of {attribute} materials",
            "I admire the boldness of {attribute} patterns",
            "I treasure the comfort of {attribute} fits",
            "I appreciate the craftsmanship of {attribute} embellishments",
            "I am inspired by the innovation of {attribute} necklines",
            "I am fond of the practicality of {attribute} sleeve lengths",
            "I delight in the elegance of {attribute} pants lengths",
            "I relish the charm of {attribute} skirt lengths",
            "I prize the sophistication of {attribute} dress lengths",
            "I am drawn to the versatility of {attribute} shirt types",
            "I am keen on the style of {attribute} jacket types",
            "I am captivated by the beauty of {attribute} dress types",
            "I am enthusiastic about the design of {attribute} pants types",
            "I am passionate about the creativity of {attribute} styles for {occasion}",
            "I am enchanted by the artistry of {attribute} materials for {occasion}",
            "I am charmed by the vibrancy of {attribute} colors for {occasion}",
            "I am thrilled by the intricacy of {attribute} patterns for {occasion}",
            "I am impressed by the functionality of {attribute} fits for {occasion}",
            "I am fascinated by the elegance of {attribute} embellishments for {occasion}",
            "I am intrigued by the timelessness of {attribute} details for {occasion}",
            "I am mesmerized by the allure of {attribute} designs",
            "I am drawn to the charm of {attribute} styles",
            "I am captivated by the richness of {attribute} materials",
            "I am fascinated by the intricacy of {attribute} patterns",
            "I am inspired by the elegance of {attribute} fits",
            "I am enchanted by the beauty of {attribute} embellishments",
            "I am impressed by the uniqueness of {attribute} necklines",
            "I am fond of the versatility of {attribute} sleeve lengths",
            "I delight in the practicality of {attribute} pants lengths",
            "I relish the sophistication of {attribute} skirt lengths",
            "I prize the timelessness of {attribute} dress lengths",
            "I am drawn to the creativity of {attribute} shirt types",
            "I am keen on the innovation of {attribute} jacket types",
            "I am captivated by the artistry of {attribute} dress types",
            "I am enthusiastic about the boldness of {attribute} pants types",
            "I am passionate about the elegance of {attribute} styles for {occasion}",
            "I am enchanted by the charm of {attribute} materials for {occasion}",
            "I am charmed by the vibrancy of {attribute} colors for {occasion}",
            "I am thrilled by the uniqueness of {attribute} patterns for {occasion}",
            "I am impressed by the functionality of {attribute} fits for {occasion}",
            "I am fascinated by the craftsmanship of {attribute} embellishments for {occasion}",
            "I am intrigued by the timelessness of {attribute} details for {occasion}"
            ],
            PreferenceType.NEGATIVE: [
            "I don't like {attribute} {item}",
            "Avoid {attribute} materials",
            "Not a fan of {attribute} colors"
            "I dislike {attribute} {item}s",
            "Not a fan of {attribute} styles",
            "I avoid {attribute} designs",
            "I don't appreciate {attribute} materials",
            "I steer clear of {attribute} patterns",
            "I don't prefer {attribute} fits",
            "I dislike {attribute} embellishments",
            "I avoid {attribute} necklines",
            "I don't like {attribute} sleeve lengths",
            "I steer clear of {attribute} pants lengths",
            "I dislike {attribute} skirt lengths",
            "I avoid {attribute} dress lengths",
            "I don't prefer {attribute} shirt types",
            "I dislike {attribute} jacket types",
            "I avoid {attribute} dress types",
            "I don't like {attribute} pants types",
            "I dislike {attribute} styles for {occasion}",
            "I avoid {attribute} materials for {occasion}",
            "I don't prefer {attribute} colors for {occasion}",
            "I dislike {attribute} patterns for {occasion}",
            "I avoid {attribute} fits for {occasion}",
            "I don't appreciate {attribute} embellishments for {occasion}",
            "I steer clear of {attribute} details for {occasion}",
            "I dislike the look of {attribute} styles",
            "I avoid the feel of {attribute} materials",
            "I don't like the appearance of {attribute} patterns",
            "I steer clear of the fit of {attribute} items",
            "I dislike the design of {attribute} embellishments",
            "I avoid the cut of {attribute} necklines",
            "I don't prefer the length of {attribute} sleeves",
            "I dislike the style of {attribute} pants lengths",
            "I avoid the shape of {attribute} skirt lengths",
            "I don't like the fit of {attribute} dress lengths",
            "I steer clear of the type of {attribute} shirts",
            "I dislike the design of {attribute} jackets",
            "I avoid the style of {attribute} dresses",
            "I don't prefer the cut of {attribute} pants",
            "I dislike the look of {attribute} styles for {occasion}",
            "I avoid the feel of {attribute} materials for {occasion}",
            "I don't like the appearance of {attribute} colors for {occasion}",
            "I steer clear of the patterns of {attribute} for {occasion}",
            "I dislike the fit of {attribute} items for {occasion}",
            "I avoid the design of {attribute} embellishments for {occasion}",
            "I don't appreciate the details of {attribute} for {occasion}",
            "I steer clear of the fit of {attribute} for {occasion}",
            "I dislike the look of {attribute} designs",
            "I avoid the feel of {attribute} materials",
            "I don't appreciate the craftsmanship of {attribute} embellishments",
            "I steer clear of the intricacy of {attribute} patterns",
            "I dislike the boldness of {attribute} colors",
            "I avoid the uniqueness of {attribute} designs",
            "I don't prefer the versatility of {attribute} fits",
            "I dislike the innovation of {attribute} materials",
            "I avoid the elegance of {attribute} necklines",
            "I don't like the practicality of {attribute} sleeve lengths",
            "I steer clear of the charm of {attribute} skirt lengths",
            "I dislike the timelessness of {attribute} dress lengths",
            "I avoid the creativity of {attribute} shirt types",
            "I don't prefer the artistry of {attribute} jacket types",
            "I dislike the sophistication of {attribute} dress types",
            "I avoid the boldness of {attribute} pants types",
            "I don't like the elegance of {attribute} styles for {occasion}",
            "I steer clear of the charm of {attribute} materials for {occasion}",
            "I dislike the vibrancy of {attribute} colors for {occasion}",
            "I avoid the intricacy of {attribute} patterns for {occasion}",
            "I don't prefer the functionality of {attribute} fits for {occasion}",
            "I dislike the craftsmanship of {attribute} embellishments for {occasion}",
            "I avoid the timelessness of {attribute} details for {occasion}",
            "I can't stand {attribute} {item}s",
            "I strongly dislike {attribute} designs",
            "I find {attribute} materials unappealing",
            "I have an aversion to {attribute} patterns",
            "I detest {attribute} fits",
            "I loathe {attribute} embellishments",
            "I can't tolerate {attribute} necklines",
            "I dislike the texture of {attribute} materials",
            "I avoid the appearance of {attribute} colors",
            "I find {attribute} styles unattractive",
            "I don't enjoy the feel of {attribute} fabrics",
            "I steer clear of {attribute} combinations",
            "I find {attribute} details off-putting",
            "I dislike the overall look of {attribute} items",
            "I avoid the use of {attribute} in clothing",
            "I don't appreciate the design of {attribute} features",
            "I find {attribute} patterns overwhelming",
            "I dislike the aesthetic of {attribute} styles",
            "I avoid the presence of {attribute} in my wardrobe",
            "I find {attribute} elements undesirable",
            "I don't like the vibe of {attribute} designs",
            "I steer clear of the influence of {attribute} in fashion",
            "I find {attribute} accents unappealing",
            "I dislike the incorporation of {attribute} in outfits",
            "I avoid the trend of {attribute} styles",
            "I don't appreciate the popularity of {attribute} patterns",
            "I find {attribute} features unattractive",
            "I dislike the prominence of {attribute} in clothing",
            "I avoid the association of {attribute} with my style",
            "I find {attribute} choices distasteful",
            "I don't like the representation of {attribute} in fashion",
            "I steer clear of the emphasis on {attribute} in designs",
            "I find {attribute} motifs unappealing",
            "I dislike the focus on {attribute} in apparel",
            "I avoid the reliance on {attribute} in outfits",
            "I don't appreciate the use of {attribute} in accessories",
            "I find {attribute} themes unattractive",
            "I dislike the integration of {attribute} in clothing lines",
            "I avoid the influence of {attribute} in my wardrobe",
            "I find {attribute} styles uninspiring",
            "I don't like the application of {attribute} in fashion",
            "I steer clear of the trend of {attribute} in designs",
            "I find {attribute} patterns unflattering",
            "I dislike the adoption of {attribute} in modern styles",
            "I avoid the incorporation of {attribute} in my outfits",
            "I don't appreciate the emphasis on {attribute} in trends",
            "I find {attribute} elements unattractive in clothing",
            "I dislike the association of {attribute} with certain styles",
            "I avoid the use of {attribute} in my personal fashion",
            "I find {attribute} features unappealing in apparel",
            "I don't like the prevalence of {attribute} in designs",
            "I steer clear of the influence of {attribute} in my clothing choices"
            ],
            PreferenceType.COMPARISON: [
            "Prefer {attribute1} over {attribute2}",
            "{attribute1} is better than {attribute2}",
            "Choose {attribute1} instead of {attribute2}",
            "I like {attribute1} more than {attribute2}",
            "I prefer {attribute1} to {attribute2}",
            "I find {attribute1} more appealing than {attribute2}",
            "I enjoy {attribute1} rather than {attribute2}",
            "I favor {attribute1} over {attribute2}",
            "I would choose {attribute1} instead of {attribute2}",
            "I think {attribute1} is superior to {attribute2}",
            "I like {attribute1} better than {attribute2}",
            "I prefer {attribute1} to {attribute2} in my outfits",
            "I find {attribute1} more suitable than {attribute2} for this occasion",
            "I enjoy {attribute1} rather than {attribute2} styles",
            "I favor {attribute1} over {attribute2} materials",
            "I would choose {attribute1} instead of {attribute2} colors",
            "I think {attribute1} is better than {attribute2} patterns",
            "I like {attribute1} more than {attribute2} fits",
            "I prefer {attribute1} to {attribute2} embellishments",
            "I find {attribute1} more appealing than {attribute2} necklines",
            "I enjoy {attribute1} rather than {attribute2} sleeve lengths",
            "I favor {attribute1} over {attribute2} pants lengths",
            "I would choose {attribute1} instead of {attribute2} skirt lengths",
            "I think {attribute1} is superior to {attribute2} dress lengths",
            "I like {attribute1} better than {attribute2} shirt types",
            "I prefer {attribute1} to {attribute2} jacket types",
            "I find {attribute1} more suitable than {attribute2} dress types",
            "I enjoy {attribute1} rather than {attribute2} pants types",
            "I favor {attribute1} over {attribute2} styles for {occasion}",
            "I would choose {attribute1} instead of {attribute2} materials for {occasion}",
            "I think {attribute1} is better than {attribute2} colors for {occasion}",
            "I like {attribute1} more than {attribute2} patterns for {occasion}",
            "I prefer {attribute1} to {attribute2} fits for {occasion}",
            "I find {attribute1} more appealing than {attribute2} embellishments for {occasion}",
            "I enjoy {attribute1} rather than {attribute2} details for {occasion}",
            "I favor {attribute1} over {attribute2} designs",
            "I would choose {attribute1} instead of {attribute2} styles",
            "I think {attribute1} is superior to {attribute2} materials",
            "I like {attribute1} better than {attribute2} patterns",
            "I prefer {attribute1} to {attribute2} fits",
            "I find {attribute1} more appealing than {attribute2} embellishments",
            "I enjoy {attribute1} rather than {attribute2} necklines",
            "I favor {attribute1} over {attribute2} sleeve lengths",
            "I would choose {attribute1} instead of {attribute2} pants lengths",
            "I think {attribute1} is superior to {attribute2} skirt lengths",
            "I like {attribute1} better than {attribute2} dress lengths",
            "I prefer {attribute1} to {attribute2} shirt types",
            "I find {attribute1} more suitable than {attribute2} jacket types",
            "I enjoy {attribute1} rather than {attribute2} dress types",
            "I favor {attribute1} over {attribute2} pants types",
            "I would choose {attribute1} instead of {attribute2} styles for {occasion}",
            "I think {attribute1} is better than {attribute2} materials for {occasion}",
            "I like {attribute1} more than {attribute2} colors for {occasion}",
            "I prefer {attribute1} to {attribute2} patterns for {occasion}",
            "I find {attribute1} more appealing than {attribute2} fits for {occasion}",
            "I enjoy {attribute1} rather than {attribute2} embellishments for {occasion}",
            "I favor {attribute1} over {attribute2} details for {occasion}",
            "I would choose {attribute1} instead of {attribute2} designs",
            "I think {attribute1} is superior to {attribute2} styles",
            "I like {attribute1} better than {attribute2} materials",
            "I prefer {attribute1} to {attribute2} patterns",
            "I find {attribute1} more appealing than {attribute2} fits",
            "I enjoy {attribute1} rather than {attribute2} embellishments",
            "I favor {attribute1} over {attribute2} necklines",
            "I would choose {attribute1} instead of {attribute2} sleeve lengths",
            "I think {attribute1} is superior to {attribute2} pants lengths",
            "I like {attribute1} better than {attribute2} skirt lengths",
            "I prefer {attribute1} to {attribute2} dress lengths",
            "I find {attribute1} more suitable than {attribute2} shirt types",
            "I enjoy {attribute1} rather than {attribute2} jacket types",
            "I favor {attribute1} over {attribute2} dress types",
            "I would choose {attribute1} instead of {attribute2} pants types",
            "I think {attribute1} is better than {attribute2} styles for {occasion}",
            "I like {attribute1} more than {attribute2} materials for {occasion}",
            "I prefer {attribute1} to {attribute2} patterns for {occasion}",
            "I find {attribute1} more appealing than {attribute2} fits for {occasion}",
            "I enjoy {attribute1} rather than {attribute2} embellishments for {occasion}",
            "I favor {attribute1} over {attribute2} details for {occasion}",
            "I would choose {attribute1} instead of {attribute2} designs for {occasion}",
            "I think {attribute1} is superior to {attribute2} styles for {occasion}",
            "I like {attribute1} better than {attribute2} materials for {occasion}",
            "I prefer {attribute1} to {attribute2} colors for {occasion}",
            "I find {attribute1} more appealing than {attribute2} patterns for {occasion}",
            "I enjoy {attribute1} rather than {attribute2} fits for {occasion}",
            "I favor {attribute1} over {attribute2} embellishments for {occasion}",
            "I would choose {attribute1} instead of {attribute2} details for {occasion}",
            "I think {attribute1} is better than {attribute2} designs for {occasion}",
            "I like {attribute1} more than {attribute2} styles for {occasion}",
            "I prefer {attribute1} to {attribute2} materials for {occasion}",
            "I find {attribute1} more appealing than {attribute2} colors for {occasion}",
            "I enjoy {attribute1} rather than {attribute2} patterns for {occasion}",
            "I favor {attribute1} over {attribute2} fits for {occasion}",
            "I would rather have {attribute1} than {attribute2}",
            "For me, {attribute1} stands out compared to {attribute2}",
            "I think {attribute1} works better than {attribute2}",
            "I feel {attribute1} is more stylish than {attribute2}",
            "In my opinion, {attribute1} is preferable to {attribute2}",
            "I find {attribute1} more versatile than {attribute2}",
            "I believe {attribute1} is more suitable than {attribute2}",
            "I lean towards {attribute1} over {attribute2}",
            "I am more inclined to choose {attribute1} than {attribute2}",
            "I consider {attribute1} to be a better choice than {attribute2}",
            "I think {attribute1} complements my style better than {attribute2}",
            "I find {attribute1} more appealing for this occasion than {attribute2}",
            "I would select {attribute1} over {attribute2} any day",
            "I feel {attribute1} is a better match for my preferences than {attribute2}",
            "I think {attribute1} has more charm than {attribute2}",
            "I find {attribute1} more practical than {attribute2}",
            "I believe {attribute1} is more fashionable than {attribute2}",
            "I think {attribute1} is more unique than {attribute2}",
            "I find {attribute1} more comfortable than {attribute2}",
            "I would always pick {attribute1} instead of {attribute2}"
            ],
            PreferenceType.NEGATION: [
            "No {attribute} please",
            "Without any {attribute}",
            "I never want {attribute} in my clothes"
            "I don't want {attribute} in my wardrobe",
            "Please exclude {attribute} from my options",
            "I prefer items without {attribute}",
            "No {attribute} designs for me",
            "I avoid anything with {attribute}",
            "I dislike {attribute} and won't consider it",
            "Keep {attribute} out of my choices",
            "I am not interested in {attribute} styles",
            "I steer clear of {attribute} features",
            "I don't appreciate {attribute} in clothing",
            "I would rather not have {attribute}",
            "I am against {attribute} in my outfits",
            "I reject {attribute} as an option",
            "I am not fond of {attribute} and avoid it",
            "I do not favor {attribute} in my fashion",
            "I am not looking for {attribute} items",
            "I am uninterested in {attribute} designs",
            "I do not like {attribute} and won't choose it",
            "I am not a fan of {attribute} and avoid it",
            "I do not want {attribute} in my style"
            "I don't care for {attribute}",
            "I am not into {attribute}",
            "I would rather avoid {attribute}",
            "I am not a fan of {attribute}",
            "I dislike the idea of {attribute}",
            "I don't want to see {attribute} in my options",
            "I am not interested in anything {attribute}",
            "I prefer to exclude {attribute} from my choices",
            "I don't feel comfortable with {attribute}",
            "I am not drawn to {attribute}",
            "I don't think {attribute} suits me",
            "I would like to avoid {attribute} altogether",
            "I am not looking for anything {attribute}",
            "I don't appreciate the presence of {attribute}",
            "I am not inclined towards {attribute}",
            "I don't find {attribute} appealing",
            "I am not open to {attribute}",
            "I don't think {attribute} works for me",
            "I am not considering {attribute} as an option",
            "I don't want {attribute} to be part of my style"
            "I don't want anything with {attribute}",
            "Please keep {attribute} out of my options",
            "I am not interested in styles featuring {attribute}",
            "I would prefer to avoid {attribute} altogether",
            "No designs with {attribute} for me",
            "I don't care for the inclusion of {attribute}",
            "I am not drawn to items with {attribute}",
            "I would rather not see {attribute} in my choices",
            "I am not a fan of clothing with {attribute}",
            "I avoid styles that incorporate {attribute}",
            "I don't like the presence of {attribute} in my wardrobe",
            "I am not looking for anything that includes {attribute}",
            "I dislike the use of {attribute} in fashion",
            "I am not open to options with {attribute}",
            "I don't appreciate the addition of {attribute} in designs",
            "I would rather exclude {attribute} from my preferences",
            "I am not fond of items that feature {attribute}",
            "I avoid trends that emphasize {attribute}",
            "I don't want to consider {attribute} as part of my style",
            "I am not inclined to choose anything with {attribute}",
            "I don't want {attribute} anywhere near my wardrobe",
            "I am completely against {attribute} in my clothing",
            "I refuse to wear anything with {attribute}",
            "I cannot stand the idea of {attribute} in my outfits",
            "I would never choose {attribute} as part of my style",
            "I strongly avoid {attribute} in my fashion choices",
            "I am not okay with {attribute} being included",
            "I absolutely dislike {attribute} in any form",
            "I am not interested in seeing {attribute} in my options",
            "I would rather not have {attribute} in my clothing",
            "I am opposed to {attribute} in my wardrobe",
            "I cannot tolerate {attribute} in my style",
            "I am not a supporter of {attribute} in fashion",
            "I would rather exclude {attribute} from my preferences",
            "I am not comfortable with {attribute} in my outfits",
            "I am not in favor of {attribute} in my clothing choices",
            "I do not want {attribute} to be part of my look",
            "I am not inclined to include {attribute} in my wardrobe",
            "I am not a fan of {attribute} and will avoid it",
            "I am not open to the idea of {attribute} in my style"
            ]
        }

    def _select_attribute(self, category: AttributeCategory) -> str:
        return random.choice(self.attribute_pool[category])

    def _generate_item(self) -> str:
        return random.choice(["shirt", "dress", "pants", "jacket", "skirt"])

    def _generate_occasion(self) -> str:
        return random.choice(self.attribute_pool[AttributeCategory.OCCASION])

    def _generate_base_sample(self) -> Dict:
        """Generate core attributes for a sample"""
        return {
            "text": "",
            "attributes": {cat.value: [] for cat in AttributeCategory},
            "sentiment": {},
            "metadata": {
                "source": "synthetic",
                "generation_type": None
            }
        }

    def _add_attribute(self, sample: Dict, category: AttributeCategory, value: str, sentiment: float):
        """Helper to add attributes with sentiment"""
        sample["attributes"][category.value].append(value)
        key = f"{category.value}:{value}"
        sample["sentiment"][key] = sentiment

    def generate_sample(self) -> Dict:
        """Generate a single synthetic sample"""
        sample = self._generate_base_sample()
        pref_type = random.choices(
            list(PreferenceType),
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]

        try:
            if pref_type == PreferenceType.POSITIVE:
                self._handle_positive(sample)
            elif pref_type == PreferenceType.NEGATIVE:
                self._handle_negative(sample)
            elif pref_type == PreferenceType.COMPARISON:
                self._handle_comparison(sample)
            else:
                self._handle_negation(sample)

            sample["metadata"]["generation_type"] = pref_type.value
            return sample
        except Exception as e:
            logger.error(f"Error generating sample: {str(e)}")
            return None

    def _handle_positive(self, sample: Dict):
        """Generate positive preference sample"""
        category = random.choice([
            AttributeCategory.COLOR,
            AttributeCategory.MATERIAL,
            AttributeCategory.STYLE
        ])
        attribute = self._select_attribute(category)
        item = self._generate_item()
        
        template = random.choice(self.templates[PreferenceType.POSITIVE])
        sample["text"] = template.format(
            attribute=attribute,
            item=item,
            occasion=self._generate_occasion()
        )
        self._add_attribute(sample, category, attribute, 0.9 + random.random()/10)

    def _handle_negative(self, sample: Dict):
        """Generate negative preference sample"""
        category = random.choice([
            AttributeCategory.MATERIAL,
            AttributeCategory.FIT,
            AttributeCategory.COLOR
        ])
        attribute = self._select_attribute(category)
        
        template = random.choice(self.templates[PreferenceType.NEGATIVE])
        sample["text"] = template.format(attribute=attribute)
        self._add_attribute(sample, category, attribute, -0.8 - random.random()/10)

    def _handle_comparison(self, sample: Dict):
        """Generate comparison preference sample"""
        category = random.choice([
            AttributeCategory.MATERIAL,
            AttributeCategory.COLOR,
            AttributeCategory.STYLE
        ])
        attrs = random.sample(self.attribute_pool[category], 2)
        
        template = random.choice(self.templates[PreferenceType.COMPARISON])
        sample["text"] = template.format(attribute1=attrs[0], attribute2=attrs[1])
        self._add_attribute(sample, category, attrs[0], 0.7 + random.random()/10)
        self._add_attribute(sample, category, attrs[1], -0.6 - random.random()/10)

    def _handle_negation(self, sample: Dict):
        """Generate negation pattern sample"""
        category = random.choice([
            AttributeCategory.MATERIAL,
            AttributeCategory.FIT
        ])
        attribute = self._select_attribute(category)
        
        template = random.choice(self.templates[PreferenceType.NEGATION])
        sample["text"] = template.format(attribute=attribute)
        self._add_attribute(sample, category, attribute, -1.0)

    def generate_dataset(self, num_samples: int = 20000) -> Dataset:
        """Generate full synthetic dataset"""
        samples = []
        while len(samples) < num_samples:
            sample = self.generate_sample()
            if sample:
                samples.append(sample)
                if len(samples) % 1000 == 0:
                    logger.info(f"Generated {len(samples)}/{num_samples} samples")
        return Dataset.from_list(samples)