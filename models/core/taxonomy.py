import json
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache

from .settings import REFINED_TAXONOMY_FILE
from models.utils.common import normalize_item_name

# Single source of truth for the parent categories
PARENT_CATEGORIES: List[str] = [
    # === RAW INGREDIENTS  ===
    "Raw Ingredient - Fruits",
    "Raw Ingredient - Vegetables",
    "Raw Ingredient - Herbs & Spices",
    "Raw Ingredient - Meat & Poultry",
    "Raw Ingredient - Fish & Seafood",
    "Raw Ingredient - Dairy & Eggs",
    "Raw Ingredient - Grains & Cereals",
    "Raw Ingredient - Legumes, Nuts & Seeds",
    "Raw Ingredient - Oils & Fats",

    # === PROCESSED INGREDIENTS ===
    "Processed Ingredient - Cheese",
    "Processed Ingredient - Dairy Alternatives",
    "Processed Ingredient - Flours & Powders",
    "Processed Ingredient - Canned & Preserved",

    # === BAKERY & BREAD ===
    "Bakery - Bread & Loaves",
    "Bakery - Pastries & Sweet Baked",
    "Bakery - Savory Baked Goods",
    "Bakery - Pizza & Flatbreads",

    # === PREPARED FOODS ===
    "Prepared Food - Soups & Broths",
    "Prepared Food - Salads & Cold Dishes",
    "Prepared Food - Pasta & Rice Dishes",
    "Prepared Food - Main Courses",
    "Prepared Food - Vegetarian Dishes",

    # === BEVERAGES  ===
    "Beverage - Non-Alcoholic",
    "Beverage - Alcoholic",

    # === CONDIMENTS & SEASONINGS ===
    "Condiment - Sauces & Dressings",
    "Condiment - Spreads & Jams",
    "Condiment - Seasonings & Spices",

    # === SNACKS & SWEETS ===
    "Snack - Savory",
    "Snack - Sweet",
    "Dessert - Cakes & Pastries",
    "Dessert - Frozen & Cold",
    "Dessert - Candy & Confections",

    "Miscellaneous"
]

class Taxonomy:
    """
    Compare the AI result with the known category list.
    """
    # Use the refined taxonomy file as the default
    def __init__(self, mapping_file: str = REFINED_TAXONOMY_FILE):
        try:
            with open(mapping_file, 'r') as f:
                self.mapping = json.load(f)

            self.parent_categories = list(self.mapping.keys())
            self.all_subcategories = []
            self.subcategory_to_parent = {}
            self.item_details = {}
            self.by_normalized = {}

            for parent, items in self.mapping.items():
                for entry in items:
                    if isinstance(entry, dict):
                        name = (entry.get("name") or "").strip()
                        display_name = (entry.get("display_name") or name).strip()
                        similar_items = entry.get("similar_items", []) or []
                    else:
                        name = str(entry).strip()
                        display_name = name
                        similar_items = []

                    if not name:
                        continue

                    self.all_subcategories.append(name)
                    self.subcategory_to_parent[name] = parent
                    self.item_details[name] = {
                        "display_name": display_name,
                        "similar_items": similar_items,
                        "parent_category": parent
                    }

                    self.by_normalized[normalize_item_name(name)] = name
                    if display_name:
                        self.by_normalized[normalize_item_name(display_name)] = name
        except Exception as e:
            print(f"Critical Error: Failed to load taxonomy from {mapping_file}: {e}")
            self.mapping = {}
            self.parent_categories = []
            self.all_subcategories = []
            self.subcategory_to_parent = {}

    def build_prompt_text(self) -> str:
        if not self.mapping:
            return "No categories available."

        text = ""
        for parent, items in self.mapping.items():
            text += f"\n{parent}:\n"
            names = [(it["name"] if isinstance(it, dict) else str(it)) for it in items]
            text += ", ".join(names) + "\n"
        return text

    def get_parent_for_subcategory(self, subcategory: str) -> Optional[str]:
        return self.subcategory_to_parent.get(subcategory)

    def process_ai_response(
        self,
        detected_foods: List[Dict[str, Any]],
        fuzzy_threshold: float = 0.6
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Match against the known category and geenerate validated_items and unlisted_foods.
        """
        validated_items = []
        unlisted_foods = []

        for food_item in detected_foods:
            ai_subcategory = food_item.get('subcategory', '')
            food_confidence = food_item.get('confidence', 'unknown')
            reasoning = food_item.get('reasoning', 'No reasoning provided')

            validated_item, unlisted_item = self._find_match(
                ai_subcategory, food_confidence, reasoning, fuzzy_threshold
            )

            if validated_item:
                validated_items.append(validated_item)
            elif unlisted_item:
                unlisted_foods.append(unlisted_item)

        return validated_items, unlisted_foods
    
    def get_item(self, query: str) -> Optional[Dict[str, Any]]:
        if not query:
            return None
        key = self.by_normalized.get(normalize_item_name(query))
        if not key:
            return None
        item = self.item_details.get(key, {}).copy()
        if not item:
            return None
        item["name"] = key
        item["parent_category"] = self.subcategory_to_parent.get(key)
        return item

    @lru_cache(maxsize=4096)
    def _find_match(self, ai_output: str, confidence: str, reasoning: str, threshold: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        cleaned_output = (ai_output or "").strip()
        norm = normalize_item_name(cleaned_output)

        # 1) Direct normalized lookup
        canonical = self.by_normalized.get(norm)
        if canonical:
            parent = self.get_parent_for_subcategory(canonical)
            return ({
                "parent_category": parent,
                "subcategory": canonical,
                "confidence": confidence,
                "reasoning": reasoning,
                "match_type": "exact"
            }, None)

        # 2) Fuzzy on normalized names
        norm_names = [normalize_item_name(s) for s in self.all_subcategories]
        matches = get_close_matches(norm, norm_names, n=1, cutoff=threshold)
        if matches:
            matched_norm = matches[0]
            canonical = self.by_normalized.get(matched_norm)
            if canonical:
                similarity = SequenceMatcher(None, norm, matched_norm).ratio()
                adjusted_confidence = self._adjust_confidence(confidence, similarity)
                parent = self.get_parent_for_subcategory(canonical)
                return ({
                    "parent_category": parent,
                    "subcategory": canonical,
                    "confidence": adjusted_confidence,
                    "reasoning": f"{reasoning} (Fuzzy matched from '{ai_output}', similarity: {similarity:.2f})",
                    "match_type": "fuzzy",
                    "original_ai_output": ai_output,
                    "similarity_score": similarity
                }, None)

        # 3) Word-based match (on normalized hyphenated tokens)
        words = set(re.findall(r'\w+', norm))
        if len(words) >= 2:
            for sub in self.all_subcategories:
                sub_norm = normalize_item_name(sub)
                sub_words = set(re.findall(r'\w+', sub_norm))
                if len(sub_words) >= 2 and (words.issubset(sub_words) or sub_words.issubset(words)):
                    similarity = len(words & sub_words) / max(len(words), len(sub_words))
                    if similarity >= threshold:
                        adjusted_confidence = self._adjust_confidence(confidence, similarity)
                        parent = self.get_parent_for_subcategory(sub)
                        return ({
                            "parent_category": parent,
                            "subcategory": sub,
                            "confidence": adjusted_confidence,
                            "reasoning": f"{reasoning} (Word matched from '{ai_output}', similarity: {similarity:.2f})",
                            "match_type": "word",
                            "original_ai_output": ai_output,
                            "similarity_score": similarity
                        }, None)

        # 4) No match
        return (None, {
            "ai_subcategory": ai_output,
            "confidence": confidence,
            "reasoning": reasoning
        })

    def _adjust_confidence(self, original_confidence: str, similarity: float) -> str:
        """Adjusts confidence based on fuzzy match quality."""
        confidence_map = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 1}
        original_score = confidence_map.get(original_confidence, 1)

        adjusted_score = original_score * similarity

        if adjusted_score >= 2.5: return 'high'
        if adjusted_score >= 1.5: return 'medium'
        return 'low'