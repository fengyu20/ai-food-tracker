# TBD: add seperate parameters for each model and provider

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
from models.core.classifier import classify_image
from models.core.common import load_provider_settings


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

    # TBD: check if there are overlaps with classify.py
    def evaluate_single_image(self, image_info: Dict[str, str], model_name: str, provider_name: str, temperature: float, fuzzy_threshold: int) -> Dict[str, Any]:
        image_id = image_info['image_id']
        
        try:
            print(f"Processing {image_id} with {model_name} via {provider_name}")
            
            # 1. Get the correct provider instance
            provider = get_provider(provider_name, self.api_key)
            
            # 2. Call the centralized classification function
            classification_result = classify_image(
                image_path=image_info['image_path'],
                provider=provider,
                model_name=model_name,
                taxonomy=self.taxonomy,
                temperature=temperature,
                fuzzy_threshold=fuzzy_threshold
            )

            # 3. Check for errors returned from the classifier
            if "error" in classification_result:
                raise Exception(classification_result["error"])

            # 4. Validate against ground truth
            validation_result = validate_against_ground_truth(
                classification_result,
                image_info['annotation_path'],
                self.taxonomy
            )
            
            # Extract key metrics for logging
            metrics = validation_result.get('validation_results', {}).get('accuracy_metrics', {})
            f1 = metrics.get('f1_score', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            print(f"{image_id} completed: F1={f1:.2f} P={precision:.2f} R={recall:.2f}")
            
            # 5. Return a combined result for batch analysis
            return {
                'image_id': image_id,
                'model': model_name,
                'provider': provider_name,
                'classification': classification_result,
                'validation': validation_result
            }

        except Exception as e:
            print(f"Error processing {image_id} with {model_name}: {e}")
            return {'image_id': image_id, 'model': model_name, 'provider': provider_name, 'error': str(e)}

    def run_batch_evaluation(
        self,
        models_by_provider: Dict[str, List[str]],
        max_images: int,
        max_workers: int,
        dataset_type: str,
        temperature: float,
        fuzzy_threshold: int
    ):

        print("\n--- Starting Batch Evaluation ---")
        image_pairs = self.discover_image_pairs(dataset_type, max_images)
        
        if not image_pairs:
            print("No image pairs found. Aborting evaluation.")
            return

        tasks = []
        for provider, models in models_by_provider.items():
            for model_name in models:
                for image_info in image_pairs:
                    tasks.append({
                        "image_info": image_info,
                        "model_name": model_name,
                        "provider_name": provider,
                        "temperature": temperature,
                        "fuzzy_threshold": fuzzy_threshold
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
        
        # Filter out errored results for analysis, but keep them for the raw log
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            print("No successful results to analyze.")
            return

        # Create a DataFrame indexed by image_id
        df_data = {}
        for res in successful_results:
            image_id = res['image_id']
            if image_id not in df_data:
                # Add ground truth and image path once per image
                gt_classes = res['validation'].get('ground_truth_classes', [])
                df_data[image_id] = {
                    'image_path': res['classification']['image_path'],
                    'ground_truth_items': ', '.join(sorted(gt_classes))
                }
            
            model_name = res['model']
            detected_items = res['classification'].get('detected_items', [])
            detected_subcategories = [item['subcategory'] for item in detected_items]
            
            # Store results for this model
            df_data[image_id][f'{model_name}_detected_items'] = ', '.join(sorted(detected_subcategories))
            df_data[image_id][f'{model_name}_f1_score'] = res['validation']['validation_results']['accuracy_metrics']['f1_score']
            df_data[image_id][f'{model_name}_precision'] = res['validation']['validation_results']['accuracy_metrics']['precision']
            df_data[image_id][f'{model_name}_recall'] = res['validation']['validation_results']['accuracy_metrics']['recall']
            df_data[image_id][f'{model_name}_latency_ms'] = res['classification']['latency_ms']
            df_data[image_id][f'{model_name}_raw_response'] = json.dumps(res['classification']['raw_response'])

        # Convert the dictionary to a DataFrame
        report_df = pd.DataFrame.from_dict(df_data, orient='index')
        
        # Reorder columns for better readability
        cols = ['image_path', 'ground_truth_items']
        models = sorted(list(set(res['model'] for res in successful_results)))
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

        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = self.output_dir / f"evaluation_report_{timestamp}.csv"
        report_df.to_csv(report_filename)
        
        print(f"\nComparative analysis report saved to: {report_filename}")

        # Save raw JSON results log
        raw_log_filename = self.output_dir / f"evaluation_raw_log_{timestamp}.json"
        with open(raw_log_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Raw results log saved to: {raw_log_filename}")


def main():
    """Main function to run the batch evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Run batch evaluation for food classification models.")

    provider_settings = load_provider_settings()
    all_models = list(itertools.chain.from_iterable(provider_settings.values()))
    all_providers = []
    for provider, models in provider_settings.items():
        all_providers.extend([provider] * len(models))
    
    parser.add_argument("--validation_dir", default="content/extracted_food_recognition/validation", help="Path to the validation dataset directory.")
    parser.add_argument("--training_dir", default="content/extracted_food_recognition/training", help="Path to the training dataset directory.")
    parser.add_argument("--output_dir", default="output/evaluation_reports", help="Directory to save evaluation reports.")
    parser.add_argument("--dataset_type", default="validation", choices=['validation', 'training'], help="Which dataset to use for evaluation.")
    
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process. If None, all images are used.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel threads.")

    # Model and Provider Configuration
    # TBD: load the provider and model from the json file

    parser.add_argument("--models", nargs='+', default=all_models, help="List of model names to evaluate.")
    parser.add_argument("--providers", nargs='+', default=all_providers, help="List of providers corresponding to the models.")
    parser.add_argument("--api_key", help="API key, if required by any of the providers.")

    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature for the models.")
    parser.add_argument("--fuzzy_threshold", type=int, default=90, help="Fuzzy matching threshold for taxonomy.")

    args = parser.parse_args()

    if len(args.models) != len(args.providers):
        raise ValueError("The number of models must match the number of providers.")

    models_by_provider = {}
    for provider, model in zip(args.providers, args.models):
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model)
        
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
        temperature=args.temperature,
        fuzzy_threshold=args.fuzzy_threshold
    )


if __name__ == "__main__":
    main()