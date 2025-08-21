import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, Set, List, Tuple, Optional
from models.core.taxonomy import Taxonomy
from models.utils.common import normalize_item_name


VALIDATION_FUZZY_THRESHOLD = 0.8  


def _string_similarity(a_norm: str, b_norm: str) -> float:
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    words_a = set(re.findall(r'\w+', a_norm))
    words_b = set(re.findall(r'\w+', b_norm))
    jacc = (len(words_a & words_b) / len(words_a | words_b)) if (words_a and words_b) else 0.0
    return max(ratio, jacc)


def _similarity_with_parent(
    pred_norm: str,
    gt_norm: str,
    pred_parent: Optional[str],
    gt_parent: Optional[str]
) -> float:
    s = _string_similarity(pred_norm, gt_norm)
    if pred_parent and gt_parent and pred_parent == gt_parent:
        s = max(s, 0.6)  # parent-category bonus for related but non-identical labels
    return s


def calculate_accuracy_metrics(predicted_set: Set[str], ground_truth_set: Set[str]) -> Dict[str, float]:
    """Calculates precision, recall, and F1 score based on two sets."""
    true_positives = len(predicted_set.intersection(ground_truth_set))
    false_positives = len(predicted_set.difference(ground_truth_set))
    false_negatives = len(ground_truth_set.difference(predicted_set))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def validate_against_ground_truth(
    classification_result: Dict[str, Any], 
    annotation_path: str, 
    taxonomy: Taxonomy,
    fuzzy_threshold: float,
    use_fuzzy_matching: bool = True
) -> Dict[str, Any]:
    try:
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)

        # Original ground truth classes: ann folder in the dataset
        ground_truth_classes = [obj['classTitle'] for obj in annotation_data.get('objects', []) if 'classTitle' in obj]
        ground_truth_classes = sorted(list(set(ground_truth_classes)))

       
        gt_items: List[Dict[str, Any]] = []
        for gt_class in ground_truth_classes:
            expanded = set()
            gt_norm = normalize_item_name(gt_class)
            expanded.add(gt_norm)

            # Include expanded set of ground truth using populate_taxonomy
            item_details = taxonomy.get_item(gt_class)
            gt_parent = None
            if item_details:
                gt_parent = item_details.get('parent_category')
                display_name = item_details.get('display_name', '')
                if display_name:
                    expanded.add(normalize_item_name(display_name))
                for s_item in item_details.get('similar_items', []):
                    if s_item:
                        expanded.add(normalize_item_name(s_item))

            # If no direct item_details, try fuzzy canonicalization of GT to a taxonomy item
            if not item_details:
                matched, _ = taxonomy._find_match(gt_class, 'unknown', 'gt', fuzzy_threshold)  
                if matched:
                    canonical = matched.get('subcategory')
                    if canonical:
                        expanded.add(normalize_item_name(canonical))
                        can_details = taxonomy.get_item(canonical)
                        if can_details:
                            gt_parent = gt_parent or can_details.get('parent_category')
                            disp = can_details.get('display_name', '')
                            if disp:
                                expanded.add(normalize_item_name(disp))
                            for s_item in can_details.get('similar_items', []):
                                if s_item:
                                    expanded.add(normalize_item_name(s_item))

            gt_items.append({
                "raw": gt_class,
                "norm": gt_norm,
                "expanded_set": expanded,
                "parent": gt_parent
            })

        if "error" in classification_result:
            predicted_subcategories: List[str] = []
        else:
            detected = classification_result.get('detected_items', [])
            unlisted = classification_result.get('unlisted_foods', [])
            
            predicted_subcategories = [item['subcategory'] for item in detected] + \
                                      [item['ai_subcategory'] for item in unlisted]

        pred_infos = []
        seen_norms = set()
        for subcat in predicted_subcategories:
            p_norm = normalize_item_name(subcat)
            if p_norm in seen_norms:
                continue
            seen_norms.add(p_norm)
            p_parent = taxonomy.get_parent_for_subcategory(subcat)
            pred_infos.append({"raw": subcat, "norm": p_norm, "parent": p_parent})

        # --- One-to-one matching with soft equivalence and fuzzy/hierarchical fallback ---
        consumed_indices = set()
        true_positives = 0
        matches_debug: List[Tuple[str, str, float]] = []

        for gt in gt_items:
            best_idx = None
            best_score = -1.0

            for idx, p in enumerate(pred_infos):
                if idx in consumed_indices:
                    continue

                # 1) Exact equivalence via expanded set (synonyms/aliases/canonical)
                if p["norm"] in gt["expanded_set"]:
                    best_idx = idx
                    best_score = 1.0
                    break

                # 2) Soft similarity (string + token + parent-category bonus), if enabled
                if use_fuzzy_matching:
                    score = _similarity_with_parent(p["norm"], gt["norm"], p["parent"], gt["parent"])
                    if score >= fuzzy_threshold and score > best_score:
                        best_idx = idx
                        best_score = score

            if best_idx is not None:
                consumed_indices.add(best_idx)
                true_positives += 1
                matches_debug.append((pred_infos[best_idx]["raw"], gt["raw"], best_score))

        false_negatives = len(gt_items) - true_positives
        false_positives = len([i for i in range(len(pred_infos)) if i not in consumed_indices])

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Provide union of expansions for transparency (not used in metric denominators)
        expanded_gt_union = set()
        for gt in gt_items:
            expanded_gt_union.update(gt["expanded_set"])

        return {
            "ground_truth_classes": ground_truth_classes,
            "expanded_ground_truth_set": sorted(list(expanded_gt_union)),
            "predicted_classes": predicted_subcategories,
            "normalized_predicted_set": sorted(list({p["norm"] for p in pred_infos})),
            "validation_results": {
                "is_correct": f1_score == 1.0,
                "accuracy_metrics": {
                    "f1_score": f1_score,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives
                },
                "matching_details": [
                    {"predicted": p, "ground_truth": g, "similarity": s} for (p, g, s) in matches_debug
                ],
                "fuzzy_threshold": fuzzy_threshold if use_fuzzy_matching else None
            }
        }

    except FileNotFoundError:
        return {"error": f"Annotation file not found: {annotation_path}", "validation_results": {}}
    except Exception as e:
        return {"error": f"An error occurred during validation: {e}", "validation_results": {}}