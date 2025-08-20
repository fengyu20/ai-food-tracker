import os
import json
from openai import OpenAI
from typing import Dict, Any
import ollama

from models.core.provider import BaseProvider
from models.core.common import load_image, clean_json_response


class LlamaProvider(BaseProvider):

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self._setup_client()

    def _setup_client(self):
        self.client = ollama.Client()
        print("Llama (Ollama) client configured.")

    def classify(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        print(f"Calling Llama API with model: {model_name}...")
        try:
            # Explicitly define the options for the API call
            options = {
                'temperature': kwargs.get('temperature', 0.1),
                'num_predict': kwargs.get('max_tokens', 2048)
            }
            
            # Ollama's support for temperature is model-dependent and might be set in the model file.
            # We pass it as an option here.
            base64_image = load_image(image_path, return_type="base64")
            
            response = self.client.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64_image]
                    }
                ],
                options=options,
                format="json"
            )
            
            raw_text = response['message']['content']
            print(f"Raw Llama Response Length: {len(raw_text)} characters")

            try:
                # The response should be a JSON object if response_format is respected
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
            print(f"Error during Local Llama classification: {e}")
            return {"error": str(e)}
