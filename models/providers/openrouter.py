import os
import json
from openai import OpenAI
from typing import Dict, Any

from models.core.provider import BaseProvider
from models.core.common import load_image, clean_json_response

class OpenRouterProvider(BaseProvider):

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self._setup_client()

    def _setup_client(self):
        self.api_key = self.api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY or pass api_key parameter.")
        
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        print(f"OpenRouter client configured.")

    def classify(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        print(f"Calling OpenRouter API with model: {model_name}...")
        try:
            # Explicitly define the parameters for the API call
            api_params = {
                'temperature': kwargs.get('temperature', 0.1),
                'max_tokens': kwargs.get('max_tokens', 2048)
            }

            base64_image = load_image(image_path, return_type="base64")

            response_args = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Respond with one valid JSON object only."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                **api_params,
            }

            try:
                # Try strict JSON mode first (many OpenRouter models support this)
                response = self.client.chat.completions.create(
                    **response_args, response_format={"type": "json_object"}
                )
            except Exception as e:
                # Fallback to normal mode if json_object not supported
                if "response_format" in str(e).lower() or "unsupported" in str(e).lower():
                    response = self.client.chat.completions.create(**response_args)
                else:
                    raise

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
