import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, Set, List, Tuple, Optional
from models.core.taxonomy import Taxonomy
from models.utils.common import normalize_item_name, food_lemma, naturalize
from models.core.semantic import SemanticFoodMatcher


VALIDATION_FUZZY_THRESHOLD = 0.8  

_CONF = {"low": 0, "medium": 1, "high": 2}
def _ci(v) -> int:
    return _CONF.get(str(v).lower(), 0) if isinstance(v, str) else 0

def _taxonomy_has(taxonomy: Taxonomy, label: str) -> bool:
    return taxonomy.get_item(label) is not None

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
                    {"predicted": p, "ground_truth": g, "score": s} for (p, g, s) in matches_debug
                ],
                "fuzzy_threshold": fuzzy_threshold if use_fuzzy_matching else None
            }
        }

    except FileNotFoundError:
        return {"error": f"Annotation file not found: {annotation_path}", "validation_results": {}}
    except Exception as e:
        return {"error": f"An error occurred during validation: {e}", "validation_results": {}}


def _expand_synonym_set(item_name: str, taxonomy: Taxonomy) -> Set[str]:
    expanded = {normalize_item_name(item_name)}
    item = taxonomy.get_item(item_name)
    parent = item.get('parent_category') if item else None
    if item:
        display_name = item.get('display_name') or ''
        if display_name:
            expanded.add(normalize_item_name(display_name))
        for similar in item.get('similar_items', []) or []:
            if similar:
                expanded.add(normalize_item_name(similar))

    # lemma alias
    lemma = food_lemma(item_name)
    if lemma:
        expanded.add(lemma)

    # base head token + taxonomy bigrams (guarded)
    normalized = normalize_item_name(item_name)
    parts = [p for p in normalized.split('-') if p]

    ADJ_TOKENS = {"dried","smoked","raw","cooked","grilled","fried","roasted","baked",
                  "boiled","steamed","sauteed","poached","mixed","sliced","chopped"}
    parts = [p for p in normalize_item_name(item_name).split('-') if p]
    if len(parts) == 2:
        a, b = parts
        expanded.add(f"{b}-{a}")  # reverse
        def singular(t): return t[:-1] if t.endswith('s') and len(t) > 3 else t
        def plural(t):   return t + 's' if not t.endswith('s') else t
        if a not in ADJ_TOKENS:
            expanded.add(f"{plural(a)}-{b}"); expanded.add(f"{singular(a)}-{b}")
        if b not in ADJ_TOKENS:
            expanded.add(f"{a}-{plural(b)}"); expanded.add(f"{a}-{singular(b)}")

    return expanded


def enhanced_validate_against_ground_truth(
    classification_result: Dict[str, Any],
    annotation_path: str,
    taxonomy: Taxonomy,
    fuzzy_threshold: float = VALIDATION_FUZZY_THRESHOLD,
    use_fuzzy_matching: bool = True,
    use_lemma: bool = True,
    use_semantic: bool = False,
    matcher=None,
    semantic_threshold: float = 0.78
) -> Dict[str, Any]:
    """Enhanced validation that adds lemma-aware matching and global pairing.

    It keeps the same output structure as validate_against_ground_truth.
    """
    try:
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)

        ground_truth_classes = [obj['classTitle'] for obj in annotation_data.get('objects', []) if 'classTitle' in obj]
        ground_truth_classes = sorted(list(set(ground_truth_classes)))

        gt_items: List[Dict[str, Any]] = []
        for gt_class in ground_truth_classes:
            expanded = _expand_synonym_set(gt_class, taxonomy)
            gt_parent = None
            item_details = taxonomy.get_item(gt_class)
            if item_details:
                gt_parent = item_details.get('parent_category')
            gt_items.append({
                "raw": gt_class,
                "norm": normalize_item_name(gt_class),
                "expanded_set": expanded,
                "lemma": food_lemma(gt_class) if use_lemma else None,
                "parent": gt_parent
            })

        # Collect predicted labels; prefer mapped taxonomy labels from unlisted items when present
        predicted_labels: List[str] = []
        if "error" not in classification_result:
            detected = classification_result.get('detected_items', []) or []
            unlisted = classification_result.get('unlisted_foods', []) or []

            # keep only medium/high confidence subcategories that are real taxonomy items
            for it in detected:
                sc = it.get('subcategory')
                if sc and _ci(it.get('confidence')) >= 1 and _taxonomy_has(taxonomy, sc):
                    predicted_labels.append(sc)

            # only keep mapped if high confidence and is a real taxonomy subcategory
            for u in unlisted:
                mapped = u.get('mapping') or u.get('mapped_item')
                if mapped and _taxonomy_has(taxonomy, mapped) and _ci(u.get('mapping_confidence')) >= 2:
                    predicted_labels.append(mapped)

                ai_guess = u.get('ai_subcategory')
                # keep ai_guess if it is a real taxonomy subcategory with >= medium confidence
                if ai_guess and _taxonomy_has(taxonomy, ai_guess) and _ci(u.get('confidence')) >= 1:
                    predicted_labels.append(ai_guess)
                # otherwise, allow only if very close to a GT class (strict, precision-safe)
                elif ai_guess and use_semantic and matcher is not None and ground_truth_classes:
                    for gt_raw in ground_truth_classes:
                        if matcher.taxonomy_similarity(ai_guess, gt_raw) >= max(semantic_threshold, 0.82):
                            predicted_labels.append(gt_raw)
                            break
                        

        # Augment composite predictions with GT-aligned semantic components (matching-only)
        if use_semantic and matcher is not None and ground_truth_classes:
            augmented: List[str] = []
            aug_thr = min(semantic_threshold, 0.70)  # slightly lower; GT-only additions
            for label in predicted_labels:
                norm_label = normalize_item_name(label)
                added = False
                if ('-with-' in norm_label or '-and-' in norm_label) or len(norm_label.split('-')) >= 2:
                    for gt_raw in ground_truth_classes:
                        s = matcher.taxonomy_similarity(label, gt_raw)
                        if s >= aug_thr:
                            augmented.append(gt_raw)
                            added = True
                if not added:
                    augmented.append(label)
            predicted_labels = augmented

        pred_infos: List[Dict[str, Any]] = []
        seen = set()
        for label in predicted_labels:
            n = normalize_item_name(label)
            if not n or n in seen:
                continue
            seen.add(n)
            pred_infos.append({
                "raw": label,
                "norm": n,
                "expanded_set": _expand_synonym_set(label, taxonomy),
                "lemma": food_lemma(label) if use_lemma else None,
                "parent": taxonomy.get_parent_for_subcategory(label)
            })

        def _gated_semantic_ok(pred_tokens: set, gt_tokens: set, p_parent, g_parent) -> bool:
            # cheap gates to reduce false positives
            if pred_tokens & gt_tokens:
                return True
            if p_parent and g_parent and p_parent == g_parent:
                return True
            return False        
        
        candidates: List[Tuple[float, str, int, int]] = []  # (score, rule, gi, pi)
        for gi, gt in enumerate(gt_items):
            for pi, p in enumerate(pred_infos):
                score = -1.0
                rule = None
                # Alias / synonym expansion both sides
                if p['norm'] in gt['expanded_set'] or gt['norm'] in p['expanded_set']:
                    score, rule = 1.0, 'alias'
                else:
                    # Lemma rule
                    if use_lemma and gt.get('lemma') and p.get('lemma') and gt['lemma'] and p['lemma'] and gt['lemma'] == p['lemma']:
                        score, rule = 0.9, 'lemma'
                    # Fuzzy + parent bonus
                    if use_fuzzy_matching and (score < fuzzy_threshold):
                        s = _similarity_with_parent(p['norm'], gt['norm'], p.get('parent'), gt.get('parent'))
                        parent_same = (p.get('parent') and gt.get('parent') and p.get('parent') == gt.get('parent'))
                        # Enforce strict threshold; no parent_same low-threshold bypass
                        if s >= fuzzy_threshold:
                            rule = 'fuzzy+parent' if parent_same else 'fuzzy'
                            candidates.append((float(s), rule, gi, pi))
                if score >= fuzzy_threshold:
                    candidates.append((float(score), rule or 'unknown', gi, pi))
                if score < fuzzy_threshold and use_semantic and matcher is not None:
                    ptoks = set(naturalize(p['raw']).split())
                    gtoks = set(naturalize(gt['raw']).split())

                    if _gated_semantic_ok(ptoks, gtoks, p.get('parent'), gt.get('parent')):
                        s = matcher.taxonomy_similarity(p['raw'], gt['raw'])
                        if s >= semantic_threshold:
                            candidates.append((float(s), 'semantic', gi, pi))

        # Greedy global matching by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_gt: Set[int] = set()
        used_pred: Set[int] = set()
        matches_debug: List[Tuple[str, str, float, str]] = []
        for score, rule, gi, pi in candidates:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            matches_debug.append((pred_infos[pi]['raw'], gt_items[gi]['raw'], float(score), rule))

        true_positives = len(matches_debug)
        false_negatives = len(gt_items) - true_positives
        false_positives = len(pred_infos) - true_positives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        expanded_gt_union = set()
        for gt in gt_items:
            expanded_gt_union.update(gt['expanded_set'])

        semantic_used = any(r == 'semantic' for (_, _, _, r) in matches_debug)
        method_label = "enhanced_lemma_semantic_matching" if semantic_used else "enhanced_lemma_matching"

        return {
            "ground_truth_classes": ground_truth_classes,
            "expanded_ground_truth_set": sorted(list(expanded_gt_union)),
            "predicted_classes": predicted_labels,
            "normalized_predicted_set": sorted(list({p['norm'] for p in pred_infos})),
            "validation_results": {
                "is_correct": f1 == 1.0,
                "accuracy_metrics": {
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives
                },
                "matching_details": [
                    {"predicted": p, "ground_truth": g, "score": s, "rule": r} for (p, g, s, r) in matches_debug
                ],
                "fuzzy_threshold": fuzzy_threshold if use_fuzzy_matching else None,
                "validation_method": method_label
            }
        }

    except FileNotFoundError:
        return {"error": f"Annotation file not found: {annotation_path}", "validation_results": {}}
    except Exception as e:
        return {"error": f"An error occurred during enhanced validation: {e}", "validation_results": {}}