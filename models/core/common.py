import json
import re
import time
from typing import Dict, Any, Tuple, List, Union
import base64
import io
from PIL import Image


def load_provider_config() -> Dict[str, Any]:
    try:
        config_path = 'models/configs/provider_config.json'
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The provider config file was not found at {config_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: The provider config file at {config_path} is not valid JSON.")
        raise


def load_image(image_path: str, max_size: int = 4096, return_type: str = "pillow") -> Union[Image.Image, str]:
    """
    Load an image, resize it if necessary, and return it in the specified format for LLM input.
    """
    try:
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"Resized image to {image.size}")

        if return_type == "pillow":
            return image
        
        elif return_type == "base64":
            # Convert PIL image to base64 string
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG") # You can change format if needed
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str

        else:
            raise ValueError(f"Unsupported return_type: {return_type}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        raise
    except Exception as e:
        print(f"Error loading image: {e}")
        raise


def clean_json_response(response_text: str) -> str:
    if not response_text or not response_text.strip():
        raise ValueError("LLM returned an empty or whitespace-only response.")
        
    cleaned = response_text.strip()

    # Remove markdown code blocks
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:-3]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:-3]

    cleaned = cleaned.strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError as e:
        print(f"Initial JSON parsing failed: {e}. Attempting to fix...")

        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)

        brace_count = 0
        last_valid_pos = 0
        in_string = False
        for i, char in enumerate(fixed):
            if char == '"' and (i == 0 or fixed[i-1] != '\\'):
                in_string = not in_string
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
        
        if last_valid_pos > 0 and last_valid_pos < len(fixed):
            fixed = fixed[:last_valid_pos]
            print("Truncated response to last valid JSON object.")

        try:
            json.loads(fixed)
            print("Successfully fixed and parsed JSON.")
            return fixed
        except json.JSONDecodeError:
            print("Could not fix JSON. Returning original cleaned string for upstream error handling.")
            return cleaned 


def create_error_response(request_id: str, start_time: float, error_message: str) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.

    Args:
        request_id: A unique identifier for the request.
        start_time: The timestamp (time.time()) when the request began.
        error_message: A descriptive message for the error.

    Returns:
        A dictionary matching the standard success response schema but for an error.
    """
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)

    return {
        "detected_items": [],
        "unlisted_foods": [],
        "primary_item": None,
        "overall_confidence": "error",
        "request_id": request_id,
        "latency_ms": latency_ms,
        "total_validated": 0,
        "total_unlisted": 0,
        "description": f"Error: {error_message}",
        "error": error_message
    }

def calculate_final_confidence(validated_items, unlisted_foods, ai_confidence):
    """Calculate final confidence considering all factors"""
    if not validated_items and not unlisted_foods:
        return 'low'
    
    confidence_scores = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 1}
    base_score = confidence_scores.get(ai_confidence, 1)
    
    if validated_items:
        item_scores = [confidence_scores.get(item['confidence'], 1) for item in validated_items]
        avg_item_score = sum(item_scores) / len(item_scores)
        
        combined_score = (base_score + avg_item_score) / 2
        
        if unlisted_foods:
            penalty = min(0.5, len(unlisted_foods) * 0.1)
            combined_score *= (1 - penalty)
    else:
        combined_score = base_score * 0.5  # Significant penalty
    
    if combined_score >= 2.5:
        return 'high'
    elif combined_score >= 1.5:
        return 'medium'
    else:
        return 'low'