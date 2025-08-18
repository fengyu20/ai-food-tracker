import os
import json
import google.generativeai as genai
from typing import Dict, Any

from models.core.provider import BaseProvider
from models.core.common import load_image, clean_json_response

class GeminiProvider(BaseProvider):

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self._setup_client()

    def _setup_client(self):
        api_key = self.api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Set GEMINI_API_KEY or pass api_key parameter.")
        
        genai.configure(api_key=api_key)
        print(f"Gemini client configured.")

    def classify(
        self,
        image_path: str,
        prompt: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        print(f"Calling Gemini API with model: {model_name}...")
        try:
            image = load_image(image_path)
            model = genai.GenerativeModel(model_name)

            response = model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=0.8,
                    max_output_tokens=2048,
                    response_mime_type="application/json" 
                )
            )
            
            raw_text = response.text
            print(f"Raw Gemini Response Length: {len(raw_text)} characters")

            cleaned_text = clean_json_response(raw_text)
            
            try:
                ai_data = json.loads(cleaned_text)
                return ai_data
            except json.JSONDecodeError as e:
                print(f"Final JSON parsing failed after cleaning: {e}")
                return {"error": "Failed to parse cleaned JSON response", "raw_response": raw_text}

        except Exception as e:
            print(f"Error during Gemini classification: {e}")
            return {"error": str(e)}
