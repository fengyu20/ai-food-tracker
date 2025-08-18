import argparse
import json

from models.core.taxonomy import Taxonomy
from models.providers import get_provider
from models.core.classifier import classify_image
from models.core.common import load_provider_settings

def main():

    provider_settings = load_provider_settings()
    available_providers = list(provider_settings.keys())
    
    parser = argparse.ArgumentParser(description="Classify a single food image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--provider", "-p", default=available_providers[0], choices=available_providers, help="Classification provider")
    parser.add_argument("--model", "-m", help="Specific model name to use (optional). If not provided, the first model listed for the provider will be used.")
    parser.add_argument("--api-key", "-k", help="API key for the provider")
    parser.add_argument("--output", "-o", help="Output file to save the JSON result")
    parser.add_argument("--fuzzy_threshold", type=int, default=90, help="Fuzzy matching threshold for taxonomy (0-100).")
    args = parser.parse_args()

    model_name = args.model
    if model_name is None:
        model_name = provider_settings[args.provider][0]
        print(f"INFO: No model specified. Using default for '{args.provider}': {model_name}")


    taxonomy = Taxonomy()
    provider = get_provider(args.provider, args.api_key)

    print(f"Classifying {args.image_path} with {model_name} via {args.provider}...")

    final_result = classify_image(
        image_path=args.image_path,
        provider=provider,
        model_name=model_name,
        taxonomy=taxonomy,
        fuzzy_threshold=args.fuzzy_threshold
    )

    if "error" in final_result:
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {final_result['error']}")
        if "raw_response" in final_result and final_result["raw_response"]:
             print(f"Raw Response: {final_result['raw_response']}")
        return

    # Remove the raw response before printing to the console for cleaner output
    final_result.pop("raw_response", None)

    result_json = json.dumps(final_result, indent=2)
    print("\n--- CLASSIFICATION RESULT ---")
    print(result_json)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result_json)
        print(f"\n Result saved to {args.output}")

if __name__ == "__main__":
    main()