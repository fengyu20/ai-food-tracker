# TBD: add display name + similar items also for the compare

import json
import random
import argparse
import time
import uuid
import itertools
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from models.core.taxonomy import Taxonomy
from models.core.validator import validate_against_ground_truth
from models.providers import get_provider
from models.core.classifier import run_classification
from models.core.common import load_provider_config


class BatchEvaluator:
    
    def __init__(self, validation_dir: str, training_dir: str, output_dir: str, api_key: str = None):
        self.validation_dir = Path(validation_dir)
        self.training_dir = Path(training_dir)
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.taxonomy = Taxonomy() # Initialize the taxonomy once for the entire batch
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print("BatchEvaluator initialized")

    def discover_image_pairs(self, dataset_type: str, max_images: int = None) -> List[Dict[str, str]]:

        base_dir = self.training_dir if dataset_type == 'training' else self.validation_dir
        img_dir = base_dir / 'img'
        ann_dir = base_dir / 'ann'
        
        image_paths = sorted(list(img_dir.glob('*.jpg')))
        
        if max_images is not None and max_images < len(image_paths):
            image_paths = random.sample(image_paths, max_images)
            
        pairs = []
        for img_path in image_paths:
            image_id = img_path.stem
            ann_path = ann_dir / f"{img_path.name}.json"
            if ann_path.exists():
                pairs.append({
                    'image_id': image_id,
                    'image_path': str(img_path),
                    'annotation_path': str(ann_path)
                })
        print(f"Discovered {len(pairs)} image/annotation pairs from '{dataset_type}' dataset.")
        return pairs


    def evaluate_single_image(self, image_info: Dict[str, str], model_name: str, provider_name: str, model_params: Dict[str, Any]) -> Dict[str, Any]:
        image_id = image_info['image_id']
        
        try:
            print(f"Processing {image_id} with {model_name} via {provider_name} using params: {model_params}")
            
            # 1. Call the centralized classification orchestrator
            classification_result = run_classification(
                image_path=image_info['image_path'],
                provider_name=provider_name,
                model_name=model_name,
                taxonomy=self.taxonomy,
                api_key=self.api_key,
                **model_params 
            )

            # 2. Validate against ground truth (this will now happen even if there's an error)
            validation_result = validate_against_ground_truth(
                classification_result,
                image_info['annotation_path'],
                self.taxonomy
            )
            
            # 3. Check for errors to log them correctly
            if "error" in classification_result:
                print(f"Error processing {image_id} with {model_name}: {classification_result['error']}")
                # We still return the full structure for consistent reporting
            else:
                # Extract key metrics for logging on success
                metrics = validation_result.get('validation_results', {}).get('accuracy_metrics', {})
                f1 = metrics.get('f1_score', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                print(f"{image_id} completed: F1={f1:.2f} P={precision:.2f} R={recall:.2f}")
            
            # 4. Return a combined result for batch analysis
            return {
                'image_id': image_id,
                'model': model_name,
                'provider': provider_name,
                'classification': classification_result, # Contains the error if one occurred
                'validation': validation_result
            }

        except Exception as e:
            # This is a fallback for unexpected errors outside of the classification call itself
            print(f"A critical error occurred processing {image_id} with {model_name}: {e}")
            return {
                'image_id': image_id, 
                'model': model_name, 
                'provider': provider_name, 
                'classification': {"error": f"Critical evaluation error: {str(e)}"},
                'validation': {} # No validation info if a critical error occurs
            }

    def run_batch_evaluation(
        self,
        models_by_provider: Dict[str, List[Dict]],
        max_images: int,
        max_workers: int,
        dataset_type: str,
        cli_params: Dict[str, Any],
        default_params: Dict[str, Any]
    ):

        print("\n--- Starting Batch Evaluation ---")
        image_pairs = self.discover_image_pairs(dataset_type, max_images)
        
        if not image_pairs:
            print("No image pairs found. Aborting evaluation.")
            return

        tasks = []
        for provider, models in models_by_provider.items():
            for model_config in models:
                model_name = model_config['model']
                
                # Combine parameters: CLI > model-specific > default
                params = default_params.copy()
                params.update(model_config.get('parameters', {}))
                params.update(cli_params)

                for image_info in image_pairs:
                    tasks.append({
                        "image_info": image_info,
                        "model_name": model_name,
                        "provider_name": provider,
                        "model_params": params
                    })

        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self.evaluate_single_image, **task): task for task in tasks}
            for i, future in enumerate(as_completed(future_to_task), 1):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    print(f"A task generated an exception: {task['image_info']['image_id']} with {task['model_name']}: {e}")
                finally:
                    print(f"--- Progress: {i}/{len(tasks)} tasks completed ---")
        
        print("\n--- Batch Evaluation Complete ---")
        self.analyze_results(all_results)

    def analyze_results(self, results: List[Dict[str, Any]]):

        print("Analyzing results and generating report...")
        
        if not results:
            print("No results to analyze.")
            return

        df_data = {}
        all_models = sorted(list(set(r['model'] for r in results)))

        for res in results:
            image_id = res['image_id']
            if image_id not in df_data:
                gt_classes = res['validation'].get('ground_truth_classes', [])
                df_data[image_id] = {
                    'image_path': res['classification'].get('image_path', 'unknown'),
                    'ground_truth_items': ', '.join(sorted(gt_classes)) if gt_classes else 'N/A'
                }
            
            model_name = res['model']
            classification_res = res['classification']
            
            if 'error' in classification_res:
                df_data[image_id][f'{model_name}_detected_items'] = "ERROR"
                df_data[image_id][f'{model_name}_raw_response'] = classification_res['error']
            else:
                detected_items = classification_res.get('detected_items', [])
                detected_subcategories = [item['subcategory'] for item in detected_items]
                df_data[image_id][f'{model_name}_detected_items'] = ', '.join(sorted(detected_subcategories))
                df_data[image_id][f'{model_name}_raw_response'] = json.dumps(classification_res.get('raw_response'))

            # These metrics are now always present, defaulting to 0.0 on error from validator
            metrics = res['validation'].get('validation_results', {}).get('accuracy_metrics', {})
            df_data[image_id][f'{model_name}_f1_score'] = metrics.get('f1_score', 0.0)
            df_data[image_id][f'{model_name}_precision'] = metrics.get('precision', 0.0)
            df_data[image_id][f'{model_name}_recall'] = metrics.get('recall', 0.0)
            df_data[image_id][f'{model_name}_latency_ms'] = classification_res.get('latency_ms', 0)

        # Convert the dictionary to a DataFrame
        report_df = pd.DataFrame.from_dict(df_data, orient='index')
        
        # Reorder columns for better readability
        cols = ['image_path', 'ground_truth_items']
        models = sorted(list(set(res['model'] for res in results)))
        for model in models:
            cols.extend([
                f'{model}_detected_items',
                f'{model}_f1_score',
                f'{model}_precision',
                f'{model}_recall',
                f'{model}_latency_ms',
                f'{model}_raw_response'
            ])
        
        # Ensure all columns exist, fill missing with None
        report_df = report_df.reindex(columns=cols)

        # Add a final row with the average of all numeric columns for easy comparison
        report_df.loc['--- AVERAGES ---'] = report_df.mean(numeric_only=True)

        # --- Generate Summary Report ---
        summary_data = []
        total_images = len(report_df.index) - 1 # Exclude average row

        for model in models:
            error_count = report_df[f'{model}_detected_items'].str.contains("ERROR", na=False).sum()
            success_count = total_images - error_count
            
            summary_data.append({
                "Model": model,
                "Average F1 Score": report_df[f'{model}_f1_score'].mean(),
                "Average Precision": report_df[f'{model}_precision'].mean(),
                "Average Recall": report_df[f'{model}_recall'].mean(),
                "Average Latency (ms)": report_df[f'{model}_latency_ms'].mean(),
                "Success Rate (%)": (success_count / total_images) * 100 if total_images > 0 else 0,
                "Total Images": total_images
            })

        summary_df = pd.DataFrame(summary_data)
        # Save the reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        detailed_filename = self.output_dir / f"evaluation_report_detailed_{timestamp}.csv"
        summary_filename = self.output_dir / f"evaluation_summary_{timestamp}.csv"
        
        report_df.to_csv(detailed_filename)
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"\nComparative analysis report saved to: {detailed_filename}")
        print(f"Summary report saved to: {summary_filename}")
        # Save raw JSON results log
        raw_log_filename = self.output_dir / f"evaluation_raw_log_{timestamp}.json"
        with open(raw_log_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Raw results log saved to: {raw_log_filename}")


def main():
    """Main function to run the batch evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Run batch evaluation for food classification models.")

    config = load_provider_config()
    all_providers_available = list(config['providers'].keys())
    
    parser.add_argument("--validation_dir", default="content/extracted_food_recognition/validation", help="Path to the validation dataset directory.")
    parser.add_argument("--training_dir", default="content/extracted_food_recognition/training", help="Path to the training dataset directory.")
    parser.add_argument("--output_dir", default="output/evaluation_reports", help="Directory to save evaluation reports.")
    parser.add_argument("--dataset_type", default="validation", choices=['validation', 'training'], help="Which dataset to use for evaluation.")
    
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process. If None, all images are used.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel threads.")

    # Simplified provider selection
    parser.add_argument("--providers", nargs='+', default=all_providers_available, help="List of providers to evaluate. If not specified, all available providers will be used.")
    parser.add_argument("--api_key", help="API key, if required by any of the providers.")

    parser.add_argument("--temperature", type=float, help="Default generation temperature. Overrides config file.")
    parser.add_argument("--fuzzy_threshold", type=int, help="Default fuzzy matching threshold. Overrides config file.")
    parser.add_argument("--max_tokens", type=int, help="Default max output tokens. Overrides config file.")

    args = parser.parse_args()

    # Build the list of models to evaluate based on the selected providers
    models_by_provider = {}
    provider_configs = config['providers']

    for provider in args.providers:
        if provider in provider_configs:
            model_list = []
            for item in provider_configs[provider]:
                if isinstance(item, str):
                    model_list.append({'model': item, 'parameters': {}})
                elif isinstance(item, dict):
                    model_list.append(item)
            models_by_provider[provider] = model_list
        else:
            print(f"WARNING: Provider '{provider}' not found in provider_config.json. Skipping.")

    if not models_by_provider:
        raise ValueError("No valid providers selected or no models found for the selected providers.")
        
    # Collect CLI parameters that can override configs. Filter out None values.
    cli_params = {
        "temperature": args.temperature,
        "fuzzy_threshold": args.fuzzy_threshold,
        "max_tokens": args.max_tokens
    }
    cli_params = {k: v for k, v in cli_params.items() if v is not None}

    evaluator = BatchEvaluator(
        validation_dir=args.validation_dir,
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        api_key=args.api_key
    )

    evaluator.run_batch_evaluation(
        models_by_provider=models_by_provider,
        max_images=args.max_images,
        max_workers=args.max_workers,
        dataset_type=args.dataset_type,
        cli_params=cli_params,
        default_params=config.get('default_parameters', {})
    )


if __name__ == "__main__":
    main()