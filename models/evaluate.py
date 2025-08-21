# TBD: continue to use openrouter to test more images to see if we can run in a batch
# check the reason, gemini works for a single image but not in a batch?
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
from models.core.validator import validate_against_ground_truth, VALIDATION_FUZZY_THRESHOLD
from models.providers import get_provider
from models.core.classifier import run_classification
from models.core.common import load_provider_config
from models.configs.config import EvaluationConfig


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


    def evaluate_single_image(self, image_info: Dict[str, str], model_name: str, provider_name: str, model_params: Dict[str, Any], use_fuzzy_matching: bool) -> Dict[str, Any]:
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
            fuzzy_threshold = model_params.get('fuzzy_threshold', VALIDATION_FUZZY_THRESHOLD)
            if fuzzy_threshold > 1.0:
                fuzzy_threshold = fuzzy_threshold / 100.0
                
            validation_result = validate_against_ground_truth(
                classification_result,
                image_info['annotation_path'],
                self.taxonomy,
                fuzzy_threshold=fuzzy_threshold,
                use_fuzzy_matching=use_fuzzy_matching
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
        default_params: Dict[str, Any],
        use_fuzzy_matching: bool
    ):

        print("\n--- Starting Batch Evaluation ---")
        image_pairs = self.discover_image_pairs(dataset_type, max_images)
        
        if not image_pairs:
            print("No image pairs found. Aborting evaluation.")
            return

        # --- Temporary file for streaming results ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_results_path = self.output_dir / f"temp_results_{timestamp}.jsonl"

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
                        "model_params": params,
                        "use_fuzzy_matching": use_fuzzy_matching
                    })

        with temp_results_path.open('w') as f_out:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(self.evaluate_single_image, **task): task for task in tasks}
                for i, future in enumerate(as_completed(future_to_task), 1):
                    try:
                        result = future.result()
                        # Write result as a JSON line to the temporary file
                        f_out.write(json.dumps(result) + '\n')
                    except Exception as e:
                        task = future_to_task[future]
                        print(f"A task generated an exception: {task['image_info']['image_id']} with {task['model_name']}: {e}")
                    finally:
                        print(f"--- Progress: {i}/{len(tasks)} tasks completed ---")
        
        print("\n--- Batch Evaluation Complete ---")
        self.analyze_results(temp_results_path)

        # Clean up the temporary file
        temp_results_path.unlink()

    def analyze_results(self, results_path: Path):
        print("Analyzing results and generating report...")
        
        try:
            with results_path.open('r') as f:
                results = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print("No results file found to analyze.")
            return

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
                # On error, metrics are 0 and latency might be present or 0
                df_data[image_id][f'{model_name}_f1_score'] = 0.0
                df_data[image_id][f'{model_name}_precision'] = 0.0
                df_data[image_id][f'{model_name}_recall'] = 0.0
                df_data[image_id][f'{model_name}_latency_ms'] = classification_res.get('latency_ms', 0)
            else:
                detected_items = classification_res.get('detected_items', [])
                detected_subcategories = [item['subcategory'] for item in detected_items]
                df_data[image_id][f'{model_name}_detected_items'] = ', '.join(sorted(detected_subcategories))
                df_data[image_id][f'{model_name}_raw_response'] = json.dumps(classification_res.get('raw_response'))
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
        
        report_df = report_df.reindex(columns=cols)

        # --- AVERAGES ROW (based on successful runs only) ---
        averages = {}
        for model in models:
            # Identify successful runs (not errored)
            error_mask = report_df[f'{model}_detected_items'] == "ERROR"
            successful_runs = report_df[~error_mask]
            
            if not successful_runs.empty:
                for metric in ['f1_score', 'precision', 'recall', 'latency_ms']:
                    col_name = f'{model}_{metric}'
                    averages[col_name] = successful_runs[col_name].mean()
            else: # If a model failed on all images
                 for metric in ['f1_score', 'precision', 'recall', 'latency_ms']:
                    averages[f'{model}_{metric}'] = 0
        
        report_df.loc['--- AVERAGES ---'] = pd.Series(averages)

        # --- SUMMARY REPORT (separating success rate from accuracy) ---
        summary_data = []
        total_images = len(report_df.index) - 1 # Exclude average row

        for model in models:
            error_mask = report_df[f'{model}_detected_items'].iloc[:-1] == "ERROR"
            successful_runs = report_df.iloc[:-1][~error_mask]
            
            success_count = len(successful_runs)
            
            summary_item = {
                "Model": model,
                "Success Rate (%)": (success_count / total_images) * 100 if total_images > 0 else 0,
                "Successful Evals": f"{success_count}/{total_images}",
                "Avg F1 (on success)": successful_runs[f'{model}_f1_score'].mean() if success_count > 0 else 0,
                "Avg Precision (on success)": successful_runs[f'{model}_precision'].mean() if success_count > 0 else 0,
                "Avg Recall (on success)": successful_runs[f'{model}_recall'].mean() if success_count > 0 else 0,
                "Avg Latency (on success, ms)": successful_runs[f'{model}_latency_ms'].mean() if success_count > 0 else 0,
            }
            summary_data.append(summary_item)

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
    parser.add_argument(
        "--config", 
        default="models/configs/evaluation_config.yml", 
        help="Path to the evaluation config YAML file."
    )
    # Allow overriding specific config values via CLI
    parser.add_argument("--max_images", type=int, help="Override max_images from config.")
    parser.add_argument("--providers", nargs='+', help="Override list of providers from config.")

    args = parser.parse_args()

    # Load base configuration from YAML
    config = EvaluationConfig.from_yaml(args.config)

    # --- Override config with CLI arguments if provided ---
    if args.max_images is not None:
        config.max_images = args.max_images
    if args.providers is not None:
        config.providers = args.providers

    # Load provider-specific model configurations
    provider_model_config = load_provider_config()
    all_providers_available = list(provider_model_config['providers'].keys())
    
    # Determine which providers to use
    providers_to_run = config.providers or all_providers_available

    # Build the list of models to evaluate
    models_by_provider = {}
    provider_configs = provider_model_config['providers']
    for provider in providers_to_run:
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
        
    # CLI parameters are now just for overrides, the base comes from the config object
    cli_params = config.default_model_parameters

    evaluator = BatchEvaluator(
        validation_dir=config.validation_dir,
        training_dir=config.training_dir,
        output_dir=config.output_dir,
        api_key=config.api_key
    )

    use_fuzzy = not cli_params.get('disable_fuzzy_matching', False)

    evaluator.run_batch_evaluation(
        models_by_provider=models_by_provider,
        max_images=config.max_images,
        max_workers=config.max_workers,
        dataset_type=config.dataset_type,
        cli_params=cli_params,
        default_params=config.default_model_parameters,
        use_fuzzy_matching=use_fuzzy
    )

if __name__ == "__main__":
    main()