import os
import json
from openai import OpenAI
from typing import Dict, Any

from models.core.provider import BaseProvider
from models.core.common import clean_json_response, image_to_base64_uri

class OpenRouterProvider(BaseProvider):
    # TBD: use the json file to get the default model
    
    DEFAULT_MODEL = "google/gemini-flash-1.5"

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.api_key = self.api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY or pass api_key parameter.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        print("OpenRouter client configured.")

    def classify(
        self,
        image_path: str,
        prompt: str,
        model_name: str = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:

        model_to_use = model_name or self.DEFAULT_MODEL
        print(f"Calling OpenRouter API with model: {model_to_use}...")

        try:
            base64_image_uri = image_to_base64_uri(image_path)

            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image_uri},
                            },
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            
            raw_text = response.choices[0].message.content
            print(f"Raw OpenRouter Response Length: {len(raw_text)} characters")

            try:
                # The response should already be a valid JSON object
                ai_data = json.loads(raw_text)
                return ai_data
            except json.JSONDecodeError:
                # Fallback to cleaning if the model fails to produce perfect JSON
                cleaned_text = clean_json_response(raw_text)
                try:
                    ai_data = json.loads(cleaned_text)
                    return ai_data
                except json.JSONDecodeError as e:
                    print(f"Final JSON parsing failed after cleaning: {e}")
                    return {"error": "Failed to parse cleaned JSON response", "raw_response": raw_text}

        except Exception as e:
            print(f"Error during OpenRouter classification: {e}")
            return {"error": str(e)}
