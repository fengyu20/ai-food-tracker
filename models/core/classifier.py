import time
import uuid
from typing import Dict, Any

from models.core.taxonomy import Taxonomy
from models.core.provider import BaseProvider
from models.core.common import calculate_final_confidence
from models.core.prompts import FOOD_DETECTION_PROMPT
from models.providers import get_provider


def run_classification(
    image_path: str,
    provider_name: str,
    model_name: str,
    taxonomy: Taxonomy,
    api_key: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    High-level orchestrator for running a single image classification.

    This function handles:
    1. Instantiating the correct provider.
    2. Calling the core `classify_image` function.
    """
    try:
        provider = get_provider(provider_name, api_key)
        
        classification_result = classify_image(
            image_path=image_path,
            provider=provider,
            model_name=model_name,
            taxonomy=taxonomy,
            **kwargs
        )
        return classification_result

    except Exception as e:
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "image_path": image_path,
            "model": model_name,
            "provider": provider_name
        }


def classify_image(
    image_path: str,
    provider: BaseProvider,
    model_name: str,
    taxonomy: Taxonomy,
    **kwargs
) -> Dict[str, Any]:
    """
    Classifies a single image using a given provider and model.

    This function encapsulates the core logic of:
    1. Building the prompt.
    2. Calling the provider's classification method.
    3. Processing the raw AI response.
    4. Calculating final confidence.
    5. Structuring the final result.
    """
    start_time = time.time()

    # 1. Build the prompt from the template
    subcategory_text = taxonomy.build_prompt_text()
    prompt = FOOD_DETECTION_PROMPT.format(subcategory_text=subcategory_text)

    # 2. Call the provider, passing all remaining kwargs
    raw_ai_result = provider.classify(image_path, prompt, model_name, **kwargs)
    latency_ms = int((time.time() - start_time) * 1000)

    if "error" in raw_ai_result:
        return {
            "error": raw_ai_result["error"],
            "raw_response": raw_ai_result.get("raw_response"),
            "request_id": str(uuid.uuid4()),
            "image_path": image_path,
            "model": model_name,
            "provider": provider.name,
            "latency_ms": latency_ms,
        }

    # 3. Process the response - convert percentage threshold to decimal
    raw_fuzzy = kwargs.get('fuzzy_threshold', 0.8)  # default as decimal for clarity
    fuzzy_threshold_decimal = (raw_fuzzy / 100.0) if raw_fuzzy > 1.0 else raw_fuzzy

    raw_items = raw_ai_result.get("detected_foods", []) or []

    if any(isinstance(it, dict) and (it.get('match_type') in ('exact_match', 'ai_guess') or 'item_name' in it or 'mapped_item' in it) for it in raw_items):
        validated_items, unlisted_foods = postprocess_predictions(
            raw_items,
            taxonomy=taxonomy,
            k=kwargs.get('max_items', 4),
            require_min_conf=kwargs.get('min_confidence', 'medium')
        )
    else:
        validated_items, unlisted_foods = taxonomy.process_ai_response(
            raw_items,
            fuzzy_threshold_decimal
        )

    # 4. Calculate final confidence
    final_confidence = calculate_final_confidence(
        validated_items,
        unlisted_foods,
        raw_ai_result.get("overall_confidence", "unknown")
    )

    # 5. Structure the final result, including the raw response for evaluation purposes
    return {
        "request_id": str(uuid.uuid4()),
        "image_path": image_path,
        "model": model_name,
        "provider": provider.name,
        "detected_items": validated_items,
        "unlisted_foods": unlisted_foods,
        "overall_confidence": final_confidence,
        "description": raw_ai_result.get("description", ""),
        "latency_ms": latency_ms,
        "raw_response": raw_ai_result
    } 

def _conf_score(label: str) -> int:
    return {'low': 0, 'medium': 1, 'high': 2}.get(label or 'low', 0)

def _score_to_label(score: int) -> str:
    return ['low', 'medium', 'high'][max(0, min(2, score))]

def postprocess_predictions(items, taxonomy, k=4, require_min_conf='medium', composite_first=True):
    if not items:
        return items

    processed_items = []
    unlisted_foods = []

    for item in items:
        mt = item.get('match_type')
        if mt == 'exact_match' or mt == 'exact':
            conf = item.get('confidence', 'low')
            sub = item.get('item_name') or item.get('subcategory')
            if not sub:
                continue
            processed_items.append({
                'subcategory': sub,
                'confidence': conf,
                'match_type': 'exact',
                'reasoning': item.get('reasoning', '')
            })
        elif mt == 'ai_guess' and item.get('mapped_item'):
            map_conf = item.get('mapping_confidence', 'low')
            if _conf_score(map_conf) >= _conf_score('medium'):
                base_conf = item.get('confidence', 'low')
                final_score = min(_conf_score(base_conf), _conf_score(map_conf))
                final_conf = _score_to_label(final_score)
                processed_items.append({
                    'subcategory': item['mapped_item'],
                    'confidence': final_conf,
                    'match_type': 'mapped',
                    'reasoning': f"AI guess '{item['item_name']}' mapped to '{item['mapped_item']}'"
                })
                unlisted_foods.append({
                    'ai_subcategory': item['item_name'],
                    'mapping': item['mapped_item'],
                    'confidence': base_conf,
                    'mapping_confidence': map_conf
                })
    # Deduplicate by subcategory (prefer higher conf; exact > mapped)
    best = {}
    for it in processed_items:
        sub = it['subcategory']
        prev = best.get(sub)
        if not prev:
            best[sub] = it
            continue
        s_new, s_old = _conf_score(it['confidence']), _conf_score(prev['confidence'])
        new_exact = (it['match_type'] == 'exact')
        old_exact = (prev['match_type'] == 'exact')
        if (s_new > s_old) or (s_new == s_old and new_exact and not old_exact):
            best[sub] = it

    deduped = list(best.values())

    deduped = [it for it in deduped if _conf_score(it['confidence']) >= _conf_score(require_min_conf)]

    deduped.sort(key=lambda it: (_conf_score(it['confidence']), it['match_type']=='exact'), reverse=True)
    deduped = deduped[:k]

    return deduped, unlisted_foods