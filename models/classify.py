import argparse
import json

from models.core.taxonomy import Taxonomy
from models.core.classifier import run_classification
from models.core.common import load_provider_config

def main():

    config = load_provider_config()
    provider_settings = config['providers']
    available_providers = list(provider_settings.keys())
    default_params = config.get('default_parameters', {})
    
    parser = argparse.ArgumentParser(description="Classify a single food image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--provider", "-p", default=available_providers[0], choices=available_providers, help="Classification provider")
    parser.add_argument("--model", "-m", help="Specific model name to use (optional). If not provided, the first model listed for the provider will be used.")
    parser.add_argument("--api-key", "-k", help="API key for the provider")
    parser.add_argument("--output", "-o", help="Output file to save the JSON result")
    parser.add_argument("--temperature", type=float, help="Generation temperature. Overrides config file.")
    parser.add_argument("--fuzzy_threshold", type=int, help="Fuzzy matching threshold. Overrides config file.")
    parser.add_argument("--max_tokens", type=int, help="Max output tokens. Overrides config file.")
    args = parser.parse_args()

    model_name = args.model
    if model_name is None:
        first_model_config = provider_settings[args.provider][0]
        if isinstance(first_model_config, str):
            model_name = first_model_config
        else:
            model_name = first_model_config['model']
        print(f"INFO: No model specified. Using default for '{args.provider}': {model_name}")

    # Layer parameters: config default -> model-specific -> CLI override
    params = default_params.copy()
    for item in provider_settings[args.provider]:
        if isinstance(item, dict) and item.get('model') == model_name:
            params.update(item.get('parameters', {}))
            break
            
    if args.temperature is not None:
        params['temperature'] = args.temperature
    if args.fuzzy_threshold is not None:
        params['fuzzy_threshold'] = args.fuzzy_threshold
    if args.max_tokens is not None:
        params['max_tokens'] = args.max_tokens

    taxonomy = Taxonomy()

    print(f"Classifying {args.image_path} with {model_name} via {args.provider} using params: {params}")

    final_result = run_classification(
        image_path=args.image_path,
        provider_name=args.provider,
        model_name=model_name,
        taxonomy=taxonomy,
        api_key=args.api_key,
        **params
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