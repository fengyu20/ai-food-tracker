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

    # Extract known parameters for this function, with defaults
    fuzzy_threshold = kwargs.get('fuzzy_threshold', 90)

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
    fuzzy_threshold_decimal = fuzzy_threshold / 100.0
    validated_items, unlisted_foods = taxonomy.process_ai_response(
        raw_ai_result.get("detected_foods", []),
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