# TBD: understand Jaccard similarity

import json
import os
import time
import argparse
import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

# Local imports for shared components
from models.core.taxonomy import PARENT_CATEGORIES
from models.core.provider import get_openrouter_client
from models.core.common import clean_json_response
from models.utils.common import setup_arg_parser, handle_clean_flag, load_progress, save_progress, normalize_item_name
from models.core.prompts import (
    FOOD_CHALLENGER_BATCH_PROMPT,
    FOOD_SYNTHESIZE_BATCH_PROMPT
)
from models.core.settings import (
    GENERATED_TAXONOMY_FILE,
    REFINED_TAXONOMY_FILE,
    REFINEMENT_PROGRESS_FILE,
    REFINEMENT_REPORT_FILE
)

def log_failed_response(model_name: str, response_text: str, error: str):
    """Logs the full text of a failed API response to a file for debugging."""
    log_dir = "logs/failed_responses"
    os.makedirs(log_dir, exist_ok=True)
    safe_model_name = model_name.replace('/', '_')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(log_dir, f"{safe_model_name}_{timestamp}.log")
    with open(filename, 'w') as f:
        f.write(f"--- Model: {model_name} ---\n")
        f.write(f"--- Error: {error} ---\n\n")
        f.write(response_text)
    print(f"      Logged failed response to: {filename}")


try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the 'openai' library to use this script: pip install openai")

# --- Configuration ---
CHALLENGER_MODEL = "anthropic/claude-3.5-haiku"
SYNTHESIZER_MODEL = "openai/gpt-5-mini"

# Constants are now imported from settings.py
RATE_LIMIT_DELAY_SECONDS = 1.0
BATCH_SIZE = 20
MAX_RETRIES = 3
MAX_SIMILAR_ITEMS = 6  # Quality control
DEFAULT_SIMILARITY_THRESHOLD = 0.4 # Default semantic distance threshold

# Category-specific thresholds for semantic disagreement.
# Lower threshold means it's more sensitive and more likely to trigger synthesis.
CATEGORY_THRESHOLDS = {
    'Raw Ingredient - Fruits': 0.35,
    'Raw Ingredient - Vegetables': 0.35,
    'Prepared Dishes': 0.5,
    'Packaged Goods': 0.55,
}


@dataclass
class FoodItem:
    name: str
    display_name: str
    parent_category: str
    similar_items: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "parent_category": self.parent_category,
            "similar_items": sorted(list(set(self.similar_items)))
        }

@dataclass
class RefinementResult:
    item_name: str
    base_version: Optional[Dict]
    challenger_version: Optional[Dict]
    synthesized_version: Optional[Dict]
    final_version: FoodItem
    success: bool
    errors: List[str]
    synthesis_triggered: bool
    disagreement_score: float

class TaxonomyRefiner:
    def __init__(self):
        self.client = get_openrouter_client()
        self.results: List[RefinementResult] = []
        progress_data = load_progress(REFINEMENT_PROGRESS_FILE)
        self.progress = {
            "completed_items": progress_data.get("completed_items", []),
            "failed_items": progress_data.get("failed_items", [])
        }
        self.generic_terms = {'food', 'dish', 'item', 'candy', 'sweet', 'sauce', 'dressing'}
        self.base_item_map: Dict[str, Dict] = {}
        
    
    def _calculate_disagreement_score(self, base_items: List[str], challenger_items: List[str]) -> float:
        """Calculate disagreement score using Jaccard similarity."""
        
        base_filtered = set(normalize_item_name(item) for item in base_items if item.strip()) - self.generic_terms
        challenger_filtered = set(normalize_item_name(item) for item in challenger_items if item.strip()) - self.generic_terms

        if not base_filtered and not challenger_filtered:
            return 0.0
        if not base_filtered or not challenger_filtered:
            return 1.0  # Max disagreement if one is empty

        try:
            intersection = len(base_filtered.intersection(challenger_filtered))
            union = len(base_filtered.union(challenger_filtered))

            if union == 0:
                return 0.0

            jaccard_similarity = intersection / union
            
            # Disagreement is 1 - similarity
            return 1.0 - jaccard_similarity
        except Exception as e:
            print(f"Could not calculate Jaccard similarity: {e}")
            return 1.0 # Default to max disagreement on error

    def _validate_response(self, response_data: Dict, item_name: str) -> Tuple[bool, List[str]]:
        """Enhanced validation with quality and contextual checks"""
        errors = []
        
        if not isinstance(response_data, dict):
            errors.append(f"Response is not a dictionary: {type(response_data)}")
            return False, errors
        
        # Check required fields
        if "parent_category" not in response_data:
            errors.append("Missing 'parent_category' field")
        elif response_data["parent_category"] not in PARENT_CATEGORIES:
            errors.append(f"Invalid parent_category: '{response_data['parent_category']}'")
        
        if "similar_items" not in response_data:
            errors.append("Missing 'similar_items' field")
        elif not isinstance(response_data["similar_items"], list):
            errors.append(f"'similar_items' must be a list, got {type(response_data['similar_items'])}")
        else:
            similar_items = response_data["similar_items"]
            
            # Check for self-reference (using normalization)
            normalized_item_name = normalize_item_name(item_name)
            if any(normalize_item_name(item) == normalized_item_name for item in similar_items):
                errors.append("Contains self-reference in similar_items")

            # Check for overly generic terms
            generic_found = [item for item in similar_items if normalize_item_name(item) in self.generic_terms]
            if generic_found:
                errors.append(f"Contains overly generic terms: {generic_found}")
            
            # Check for foreign terms (basic check)
            foreign_chars = set('àáâãäåèéêëìíîïñòóôõöùúûüÿß')
            for item in similar_items:
                if any(char in foreign_chars for char in item.lower()):
                    errors.append(f"Possible foreign term: '{item}'")
            
            # Check length
            if len(similar_items) > MAX_SIMILAR_ITEMS:
                errors.append(f"Too many similar_items ({len(similar_items)} > {MAX_SIMILAR_ITEMS})")
            
            # Contextual Category Validation
            proposed_category = response_data.get("parent_category")
            if proposed_category and self.base_item_map:
                for s_item in similar_items:
                    norm_s_item = normalize_item_name(s_item)
                    if norm_s_item in self.base_item_map:
                        known_category = self.base_item_map[norm_s_item]['parent_category']
                        if known_category != proposed_category:
                            errors.append(
                                f"Contextual mismatch for '{s_item}': its known category "
                                f"is '{known_category}', but item's proposed category is '{proposed_category}'"
                            )
        
        return len(errors) == 0, errors
    
    def _get_challenger_batch_response(self, batch: List[Dict]) -> Dict[str, Any]:
        """Get challenger responses for a batch of items."""
        if not batch:
            return {}

        item_review_parts = []
        for item in batch:
            item_str = (
                f"ITEM: \"{item['name']}\"\n"
                f"- Current Category: {item.get('parent_category', 'N/A')}\n"
                f"- Current Similar Items: {item.get('similar_items', [])}"
            )
            item_review_parts.append(item_str)
        
        items_to_review_str = "\n\n".join(item_review_parts)
        parent_categories_str = "\n".join([f"- {cat}" for cat in PARENT_CATEGORIES])

        prompt = FOOD_CHALLENGER_BATCH_PROMPT.format(
            parent_categories_list=parent_categories_str,
            items_to_review_str=items_to_review_str
        )

        for attempt in range(MAX_RETRIES):
            raw_response_content = ""
            try:
                response = self.client.chat.completions.create(
                    model=CHALLENGER_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a food classification expert. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                    temperature=0.1
                )
                raw_response_content = response.choices[0].message.content
                response_text = clean_json_response(raw_response_content)
                data = json.loads(response_text)
                
                # Basic validation that all items are in the response
                if all(item['name'] in data for item in batch):
                    return data
                else:
                    print(f"Challenger attempt {attempt+1} missing items in response.")

            except Exception as e:
                print(f"Challenger batch attempt {attempt + 1} failed: {e}")
                if raw_response_content:
                    log_failed_response(CHALLENGER_MODEL, raw_response_content, str(e))
                time.sleep(1)
        
        print(f"      Challenger batch failed after {MAX_RETRIES} attempts")
        return {}

    def _get_synthesizer_batch_response(self, items_for_synthesis: List[Dict]) -> Dict[str, Any]:
        """Get synthesizer responses for a batch of disagreed items."""
        if not items_for_synthesis:
            return {}

        item_synthesize_parts = []
        for item in items_for_synthesis:
            base = item['base']
            challenger = item['challenger']
            item_str = (
                f"ITEM: \"{base['name']}\"\n"
                f"EXPERT 1 (BASE):\n"
                f"- Category: {base.get('parent_category', 'Unknown')}\n"
                f"- Similar Items: {base.get('similar_items', [])}\n"
                f"EXPERT 2 (CHALLENGER):\n"
                f"- Category: {challenger.get('parent_category', 'Unknown')}\n"
                f"- Similar Items: {challenger.get('similar_items', [])}"
            )
            item_synthesize_parts.append(item_str)
            
        items_to_synthesize_str = "\n\n".join(item_synthesize_parts)
        parent_categories_str=f"{', '.join(PARENT_CATEGORIES[:3])}... ({len(PARENT_CATEGORIES)} total)"

        prompt = FOOD_SYNTHESIZE_BATCH_PROMPT.format(
            parent_categories_str=parent_categories_str,
            items_to_synthesize_str=items_to_synthesize_str
        )

        for attempt in range(MAX_RETRIES):
            raw_response_content = ""
            try:
                response = self.client.chat.completions.create(
                    model=SYNTHESIZER_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert food taxonomy synthesizer. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                    temperature=0.0
                )
                raw_response_content = response.choices[0].message.content
                response_text = clean_json_response(raw_response_content)
                data = json.loads(response_text)
                
                if all(item['base']['name'] in data for item in items_for_synthesis):
                    return data
                else:
                    print(f"Synthesizer attempt {attempt+1} missing items in response.")

            except Exception as e:
                print(f"Synthesizer batch attempt {attempt + 1} failed: {e}")
                if raw_response_content:
                    log_failed_response(SYNTHESIZER_MODEL, raw_response_content, str(e))
                time.sleep(1)

        print(f"Synthesizer batch failed after {MAX_RETRIES} attempts")
        return {}

    def create_fallback_version(self, item_name: str, base: Dict, challenger: Optional[Dict]) -> FoodItem:
        """Create fallback version when synthesis fails"""
        
        # Prefer challenger if available and valid, otherwise use base
        source = challenger if challenger else base
        
        # Extract with defaults
        category = source.get('parent_category', 'Miscellaneous')
        if category not in PARENT_CATEGORIES:
            category = 'Miscellaneous'
        
        similar_items = source.get('similar_items', [])
        if not isinstance(similar_items, list):
            similar_items = []
        
        # Clean similar items
        cleaned_similar = []
        for item in similar_items[:MAX_SIMILAR_ITEMS]:
            if isinstance(item, str) and item.lower() != item_name.lower():
                cleaned_similar.append(item.lower())
        
        return FoodItem(
            name=item_name,
            display_name=base.get('display_name', item_name), # Get display_name from base
            parent_category=category,
            similar_items=cleaned_similar
        )
    
    def refine_taxonomy(self, force: bool = False) -> bool:
        """Main refinement process with robust error handling"""
        
        print("Starting Taxonomy Refinement Process...")
        
        # Load base taxonomy
        try:
            with open(GENERATED_TAXONOMY_FILE, 'r') as f:
                base_data = json.load(f)
        except Exception as e:
            print(f"Could not load base taxonomy: {e}")
            return False
        
        # Create structured item mapping and populate base_item_map for contextual validation
        base_items = []
        for original_category, items in base_data.items():
            for item in items:
                item_data = {
                    'name': item['name'],
                    'display_name': item.get('display_name', item['name']), # Read display_name
                    'parent_category': original_category,
                    'similar_items': item.get('similar_items', [])
                }
                base_items.append(item_data)
        
        self.base_item_map = {normalize_item_name(item['name']): item for item in base_items}
        
        total_items = len(base_items)
        completed_items = set(self.progress.get("completed_items", []))

        if force:
            print("\n--force flag detected. All items will be re-processed.")
            remaining_items = base_items
        else:
            remaining_items = [item for item in base_items if item['name'] not in completed_items]
        
        print(f" Total items: {total_items}")
        print(f" Already completed: {len(completed_items) if not force else 0}")
        print(f" Remaining: {len(remaining_items)}")
        
        if not remaining_items:
            print("All items already processed!")
            return True
        
        # Process remaining items
        total_remaining = len(remaining_items)
        for i in range(0, total_remaining, BATCH_SIZE):
            batch_items = remaining_items[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n--- Processing Batch [{batch_num}/{total_batches}] ({len(batch_items)} items) ---")

            # 1. Get challenger responses for the whole batch
            challenger_batch_results = self._get_challenger_batch_response(batch_items)

            # 2. Compare, validate, and partition batch
            items_for_synthesis = []
            processed_in_batch = {}

            for base_item in batch_items:
                item_name = base_item['name']
                challenger_version = challenger_batch_results.get(item_name)
                disagreement_score = 0.0
                
                if not challenger_version:
                    processed_in_batch[item_name] = {"errors": ["Challenger model failed for this item in batch"]}
                    continue

                is_valid, validation_errors = self._validate_response(challenger_version, item_name)
                if not is_valid:
                    processed_in_batch[item_name] = {"errors": validation_errors, "challenger": challenger_version}
                    continue
                
                # Calculate disagreement score with normalization and filtering
                category_changed = challenger_version.get("parent_category") != base_item.get("parent_category")
                
                disagreement_score = self._calculate_disagreement_score(
                    base_item.get("similar_items", []),
                    challenger_version.get("similar_items", [])
                )
                
                # Use dynamic threshold for synthesis trigger
                base_category = base_item.get("parent_category")
                threshold = CATEGORY_THRESHOLDS.get(base_category, DEFAULT_SIMILARITY_THRESHOLD)
                significant_disagreement = category_changed or (disagreement_score > threshold)

                if significant_disagreement:
                    items_for_synthesis.append({"base": base_item, "challenger": challenger_version})
                    print(f"      ! Significant disagreement for '{item_name}' (Score: {disagreement_score:.2f}, Threshold: {threshold}, Category changed: {category_changed}). Flagged for synthesis.")
                    print(f"        - BASE:       Category='{base_item.get('parent_category')}', Similar={base_item.get('similar_items')}")
                    print(f"        - CHALLENGER: Category='{challenger_version.get('parent_category')}', Similar={challenger_version.get('similar_items')}")
                else:
                    processed_in_batch[item_name] = {
                        "final_data": challenger_version,
                        "challenger": challenger_version,
                        "synthesized": None,
                        "synthesis_triggered": False,
                        "disagreement_score": disagreement_score,
                        "errors": []
                    }
            
            # 3. Get synthesizer responses for the disagreed subset
            synthesizer_batch_results = self._get_synthesizer_batch_response(items_for_synthesis)

            # 4. Process synthesized results
            for item_to_synth in items_for_synthesis:
                item_name = item_to_synth['base']['name']
                base_item = item_to_synth['base']
                challenger_item = item_to_synth['challenger']
                synthesized_version = synthesizer_batch_results.get(item_name)

                # Recalculate score for reporting consistency
                score = self._calculate_disagreement_score(
                    base_item.get("similar_items", []),
                    challenger_item.get("similar_items", [])
                )
                
                if synthesized_version:
                    is_valid, validation_errors = self._validate_response(synthesized_version, item_name)
                    if is_valid:
                        processed_in_batch[item_name] = {
                            "final_data": synthesized_version,
                            "challenger": item_to_synth['challenger'],
                            "synthesized": synthesized_version,
                            "synthesis_triggered": True,
                            "disagreement_score": score,
                            "errors": []
                        }
                    else:
                        processed_in_batch[item_name] = {
                            "challenger": item_to_synth['challenger'],
                            "errors": validation_errors,
                            "synthesis_triggered": True,
                             "disagreement_score": score,
                        }
                else:
                    processed_in_batch[item_name] = {
                        "challenger": item_to_synth['challenger'],
                        "errors": ["Synthesizer model failed for this item in batch"],
                        "synthesis_triggered": True,
                        "disagreement_score": score,
                    }

            # 5. Create final result objects for the entire batch
            for base_item in batch_items:
                item_name = base_item['name']
                result_data = processed_in_batch.get(item_name, {})

                # Check if already processed (e.g., in a previous run) - removed for --force compatibility
                
                final_data = result_data.get('final_data')
                
                if final_data:
                    success = True
                    final_version = FoodItem(
                        name=item_name,
                        display_name=base_item.get('display_name', item_name),
                        parent_category=final_data['parent_category'],
                        similar_items=final_data['similar_items']
                    )
                else:
                    success = False
                    # Pass the challenger version to fallback if it exists
                    challenger_for_fallback = result_data.get('challenger')
                    final_version = self.create_fallback_version(item_name, base_item, challenger_for_fallback)
                
                result = RefinementResult(
                    item_name=item_name,
                    base_version=base_item,
                    challenger_version=result_data.get('challenger'),
                    synthesized_version=result_data.get('synthesized'),
                    final_version=final_version,
                    success=success,
                    errors=result_data.get('errors', ["Unknown processing error"]),
                    synthesis_triggered=result_data.get('synthesis_triggered', True),
                    disagreement_score=result_data.get('disagreement_score', 1.0)
                )
                self.results.append(result)

                # Update progress
                self.progress["completed_items"].append(item_name)
                if not success:
                    self.progress["failed_items"].append(item_name)

            # Save progress after each batch
            save_progress(self.progress, REFINEMENT_PROGRESS_FILE)
            self._save_intermediate_results(REFINED_TAXONOMY_FILE)
            print(f"  --- Batch Complete ---")
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
        
        # Save final results
        self._save_final_taxonomy(REFINED_TAXONOMY_FILE)
        self._generate_refinement_report()
        
        return True
    
    def _save_intermediate_results(self, output_file: str):
        """Save current results as backup"""
        backup_file = output_file.replace('.json', '_backup.json')
        try:
            final_taxonomy = self._build_final_taxonomy()
            with open(backup_file, 'w') as f:
                json.dump(final_taxonomy, f, indent=4)
            print(f"Intermediate results saved to {backup_file}")
        except Exception as e:
            print(f"Could not save intermediate results: {e}")
    
    def _build_final_taxonomy(self) -> Dict[str, List[Dict]]:
        """Build final taxonomy from results"""
        final_taxonomy = {cat: [] for cat in PARENT_CATEGORIES}
        
        for result in self.results:
            category = result.final_version.parent_category
            if category in final_taxonomy:
                final_taxonomy[category].append({
                    "name": result.final_version.name,
                    "display_name": result.final_version.display_name,
                    "similar_items": result.final_version.similar_items
                })
        
        # Sort items within each category and remove empty categories
        for category in final_taxonomy:
            final_taxonomy[category] = sorted(final_taxonomy[category], key=lambda x: x['name'])
        
        return {k: v for k, v in final_taxonomy.items() if v}
    
    def _save_final_taxonomy(self, output_file: str):
        """Save final refined taxonomy"""
        try:
            final_taxonomy = self._build_final_taxonomy()
            with open(output_file, 'w') as f:
                json.dump(final_taxonomy, f, indent=4)
            print(f"Final refined taxonomy saved to: {output_file}")
        except Exception as e:
            print(f"Could not save final taxonomy: {e}")
    
    def _generate_refinement_report(self):
        """Generate detailed refinement report"""
        report_file = REFINEMENT_REPORT_FILE
        
        try:
            with open(report_file, 'w') as f:
                f.write("TAXONOMY REFINEMENT REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                total = len(self.results)
                successful = len([r for r in self.results if r.success])
                failed = total - successful
                
                synthesized_count = len([r for r in self.results if r.synthesis_triggered and r.success])
                minor_change_count = len([r for r in self.results if not r.synthesis_triggered and r.success])

                f.write(f"SUMMARY:\n")
                f.write(f"Total items processed: {total}\n")
                f.write(f"Successful refinements: {successful} ({successful/total*100:.1f}%)\n")
                f.write(f"  - via Synthesizer (significant disagreement): {synthesized_count}\n")
                f.write(f"  - via Challenger (minor disagreement/agreement): {minor_change_count}\n")
                f.write(f"Fallback versions used: {failed} ({failed/total*100:.1f}%)\n\n")

                f.write(f"QUALITY METRICS (for {len(self.results)} processed items):\n")
                if self.results:
                    avg_score = sum(r.disagreement_score for r in self.results) / len(self.results)
                    trigger_rate = len([r for r in self.results if r.synthesis_triggered]) / len(self.results) * 100
                    f.write(f"  - Average Initial Disagreement Score: {avg_score:.2f}\n")
                    f.write(f"  - Synthesis Trigger Rate: {trigger_rate:.1f}%\n\n")
                else:
                    f.write("  - No items were processed to generate metrics.\n\n")

                # Error analysis
                all_errors = []
                for result in self.results:
                    all_errors.extend(result.errors)
                
                error_counts = Counter(all_errors)
                f.write("ERROR ANALYSIS:\n")
                for error, count in error_counts.most_common():
                    f.write(f"  {error}: {count} occurrences\n")
                f.write("\n")
                
                # Failed items
                failed_items = [r for r in self.results if not r.success]
                if failed_items:
                    f.write("ITEMS WITH FALLBACK VERSIONS:\n")
                    for result in failed_items:
                        f.write(f"  {result.item_name}: {'; '.join(result.errors)}\n")
            
            print(f" Refinement report saved to: {report_file}")
            
        except Exception as e:
            print(f"Could not generate report: {e}")

def main():
    parser = setup_arg_parser("Refine the food taxonomy using a challenger-synthesizer model.")
    args = parser.parse_args()

    backup_file = REFINED_TAXONOMY_FILE.replace('.json', '_backup.json')
    files_to_remove = [REFINEMENT_PROGRESS_FILE, backup_file, REFINEMENT_REPORT_FILE]
    handle_clean_flag(args.clean, files_to_remove)

    refiner = TaxonomyRefiner()
    success = refiner.refine_taxonomy(force=args.force)
    
    if success:
        print("\n🎯 REFINEMENT COMPLETE!")
        print(f"Refined taxonomy: {REFINED_TAXONOMY_FILE}")
        print(f"Report: {REFINEMENT_REPORT_FILE}")
    else:
        print("\nREFINEMENT FAILED!")

if __name__ == "__main__":
    main()