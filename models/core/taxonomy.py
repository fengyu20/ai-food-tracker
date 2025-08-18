import json
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional


class Taxonomy:
    """
    Compare the AI result with the known category list.
    """
    # TBD: include the python file to extract the categories from the dataset //

    def __init__(self, mapping_file: str = 'config/categories_mapping.json'):
        try:
            with open(mapping_file, 'r') as f:
                self.mapping = json.load(f)
            
            self.parent_categories = list(self.mapping.keys())
            self.all_subcategories = []
            self.subcategory_to_parent = {}

            for parent, subcategories in self.mapping.items():
                for subcategory in subcategories:
                    self.all_subcategories.append(subcategory)
                    self.subcategory_to_parent[subcategory] = parent
            
            print(f"📋 Taxonomy loaded: {len(self.parent_categories)} parent categories, {len(self.all_subcategories)} subcategories.")
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
        for parent, subcategories in self.mapping.items():
            text += f"\n{parent}:\n"
            text += ", ".join(subcategories)
            text += "\n"
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

    def _find_match(
        self,
        ai_output: str,
        confidence: str,
        reasoning: str,
        threshold: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process the AI result to find the best match.
        """
        # 1. Exact Match (case-insensitive)
        cleaned_output = ai_output.strip().lower()
        if ai_output in self.all_subcategories:
            parent = self.get_parent_for_subcategory(ai_output)
            print(f"  EXACT MATCH: {parent} -> {ai_output}")
            return ({
                "parent_category": parent,
                "subcategory": ai_output,
                "confidence": confidence,
                "reasoning": reasoning,
                "match_type": "exact"
            }, None)

        # 2. Fuzzy Match
        matches = get_close_matches(cleaned_output,
                                    [sub.lower() for sub in self.all_subcategories],
                                    n=1, cutoff=threshold)
        if matches:
            matched_lower = matches[0]
            # Find original case version
            for sub in self.all_subcategories:
                if sub.lower() == matched_lower:
                    similarity = SequenceMatcher(None, cleaned_output, matched_lower).ratio()
                    adjusted_confidence = self._adjust_confidence(confidence, similarity)
                    parent = self.get_parent_for_subcategory(sub)
                    print(f" FUZZY MATCH: {ai_output} -> {sub} (sim: {similarity:.2f})")
                    return ({
                        "parent_category": parent,
                        "subcategory": sub,
                        "confidence": adjusted_confidence,
                        "reasoning": f"{reasoning} (Fuzzy matched from '{ai_output}', similarity: {similarity:.2f})",
                        "match_type": "fuzzy",
                        "original_ai_output": ai_output,
                        "similarity_score": similarity
                    }, None)
        
        # 3. Word-based Match
        words = set(re.findall(r'\w+', cleaned_output))
        if len(words) >= 2:
            for sub in self.all_subcategories:
                sub_words = set(re.findall(r'\w+', sub.lower()))
                if len(sub_words) >= 2 and (words.issubset(sub_words) or sub_words.issubset(words)):
                    similarity = len(words & sub_words) / max(len(words), len(sub_words))
                    if similarity >= threshold:
                        adjusted_confidence = self._adjust_confidence(confidence, similarity)
                        parent = self.get_parent_for_subcategory(sub)
                        print(f" WORD MATCH: {ai_output} -> {sub} (sim: {similarity:.2f})")
                        return ({
                            "parent_category": parent,
                            "subcategory": sub,
                            "confidence": adjusted_confidence,
                            "reasoning": f"{reasoning} (Word matched from '{ai_output}', similarity: {similarity:.2f})",
                            "match_type": "word",
                            "original_ai_output": ai_output,
                            "similarity_score": similarity
                        }, None)

        # 4. No Match Found
        print(f"UNLISTED: {ai_output} (no close match found)")
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