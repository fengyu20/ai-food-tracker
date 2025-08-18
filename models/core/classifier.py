import time
import uuid
from typing import Dict, Any

from models.core.taxonomy import Taxonomy
from models.core.provider import BaseProvider
from models.core.common import calculate_final_confidence
from models.core.prompts import BASE_PROMPT_TEMPLATE

def classify_image(
    image_path: str,
    provider: BaseProvider,
    model_name: str,
    taxonomy: Taxonomy,
    temperature: float = 0.1,
    fuzzy_threshold: int = 90
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
    prompt = BASE_PROMPT_TEMPLATE.format(subcategory_text=subcategory_text)

    # 2. Call the provider
    raw_ai_result = provider.classify(image_path, prompt, model_name, temperature=temperature)
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