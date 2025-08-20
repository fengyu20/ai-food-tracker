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
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        print(f"Calling Gemini API with model: {model_name}...")
        try:
            # Build the generation config from known parameters in kwargs
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.1),
                top_p=0.8,
                max_output_tokens=kwargs.get('max_tokens', 2048),
                response_mime_type="application/json" 
            )

            image = load_image(image_path)
            model = genai.GenerativeModel(model_name)

            response = model.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            # Add a check to ensure the response has valid content before accessing .text
            if not response.parts:
                finish_reason = response.prompt_feedback.block_reason or "Unknown"
                error_message = f"Gemini API returned no content, likely due to safety settings or other filters. Finish reason: {finish_reason}"
                print(error_message)
                return {"error": error_message, "raw_response": str(response.prompt_feedback)}

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
