

import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional

from models.core.provider import get_openrouter_client
from models.core.taxonomy import PARENT_CATEGORIES
from models.core.common import clean_json_response
from models.utils.common import setup_arg_parser, handle_clean_flag, ensure_dir_exists, load_progress, save_progress
from models.core.prompts import FOOD_POPULATE_PROMPT, FOOD_NAME_SIMPLIFIER_PROMPT
from models.core.settings import (
    SOURCE_META_FILE,
    GENERATED_TAXONOMY_FILE,
    POPULATION_PROGRESS_FILE
)

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the 'openai' library to use this script: pip install openai")

# --- Configuration ---
TEACHER_MODEL = "mistralai/devstral-small-2505:free"
# test option: mistralai/mistral-small-3.2-24b-instruct, openai/gpt-oss-20b:free
# backup option:mistralai/mistral-medium-3.1

# Constants are now imported from settings.py, so we remove them from here.
RATE_LIMIT_DELAY_SECONDS = 1.0

BATCH_SIZE = 10 
MAX_TOKENS = 8192  

def create_prompt(items: List[str]) -> str:
    """Create unified prompt for single item or batch processing"""
    food_items_str = "\n".join([f"- {item}" for item in items])
    parent_categories_str = "\n".join([f"- {cat}" for cat in PARENT_CATEGORIES])
    
    return FOOD_POPULATE_PROMPT.format(
        parent_categories=parent_categories_str,
        food_items=food_items_str
    )

def get_simplified_name(client: OpenAI, item_name: str) -> str:
    """Use an LLM to generate a user-friendly display name."""
    if "-" not in item_name and "_" not in item_name:
        return item_name

    prompt = FOOD_NAME_SIMPLIFIER_PROMPT.format(item_name=item_name)
    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only responds in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=128,
            temperature=0.0
        )
        response_text = clean_json_response(response.choices[0].message.content)
        data = json.loads(response_text)
        if isinstance(data, dict) and "display_name" in data:
            print(f"      Simplification: '{item_name}' -> '{data['display_name']}'")
            return data['display_name']
    except Exception as e:
        print(f" Name simplification failed for '{item_name}': {e}")
    
    # Fallback to a simple replacement if the LLM fails
    return item_name.replace("-", " ").replace("_", " ")

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def validate_json_structure(data: dict, expected_items: List[str]) -> bool:
    """Validate that JSON response has correct structure"""
    if not isinstance(data, dict):
        return False
    
    for item in expected_items:
        if item not in data:
            print(f"    - Missing item '{item}' in response")
            return False
        
        item_data = data[item]
        if not isinstance(item_data, dict):
            print(f"    - Item '{item}' data is not a dict")
            return False
        
        if "parent_category" not in item_data or "similar_items" not in item_data:
            print(f"    - Item '{item}' missing required fields")
            return False
        
        if item_data["parent_category"] not in PARENT_CATEGORIES:
            print(f"    - Item '{item}' has invalid parent category: {item_data['parent_category']}")
            return False
        
        if not isinstance(item_data["similar_items"], list):
            print(f"    - Item '{item}' similar_items is not a list")
            return False
    
    return True

def process_single_item(client: OpenAI, item: str) -> Optional[Dict[str, Any]]:
    """Process a single food item with retry logic"""
    print(f"    - Processing single item: {item}")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            prompt = create_prompt([item])
            
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that only responds in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=2048,
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            
            if validate_json_structure(data, [item]):
                print(f"Successfully processed {item}")
                return data
            else:
                print(f"Invalid structure for {item} (attempt {attempt + 1})")
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {item} (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"Error processing {item} (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)  # Brief pause before retry
    
    print(f"Failed to process {item} after {max_retries} attempts")
    return None

def process_batch_individually(client: OpenAI, batch: List[str]) -> Dict[str, Any]:
    """Process each item in batch individually"""
    print(f"    - Processing {len(batch)} items individually...")
    result = {}
    
    for item in batch:
        item_result = process_single_item(client, item)
        if item_result:
            result.update(item_result)
        time.sleep(0.5)  # Rate limiting for individual requests
    
    return result

def process_batch(client: OpenAI, batch: List[str]) -> Dict[str, Any]:
    """Process a batch with fallback to single-item processing"""
    print(f"  > Processing batch of {len(batch)} items...")
    
    # Estimate token usage
    prompt = create_prompt(batch)
    
    estimated_input_tokens = estimate_tokens(prompt)
    estimated_output_tokens = len(batch) * 200  # ~200 tokens per item
    estimated_total = estimated_input_tokens + estimated_output_tokens
    
    print(f"    - Estimated tokens: {estimated_total} (input: {estimated_input_tokens}, output: {estimated_output_tokens})")
    
    if estimated_total > MAX_TOKENS * 0.9:  # 90% safety margin
        print(f"    - Batch too large, falling back to single-item processing")
        return process_batch_individually(client, batch)
    
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that only responds in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_TOKENS,
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            
            if validate_json_structure(data, batch):
                print(f"Successfully processed batch of {len(data)} items")
                return data
            else:
                print(f"Invalid batch structure (attempt {attempt + 1})")
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt + 1}): {e}")
            if attempt == 0:  # Show truncated response on first attempt
                print(f"    - Response length: {len(response_text)} chars")
                print(f"    - Response preview: {response_text[:500]}...")
        except Exception as e:
            print(f"Error processing batch (attempt {attempt + 1}): {e}")
    
    print(f"    - Batch processing failed, falling back to individual processing")
    return process_batch_individually(client, batch)

def save_taxonomy_backup(taxonomy: Dict, filename: str):
    """Save a backup of the generated taxonomy."""
    backup_filename = filename.replace('.json', '_backup.json')
    try:
        ensure_dir_exists(backup_filename)
        with open(backup_filename, 'w') as f:
            json.dump(taxonomy, f, indent=4)
        print(f"    - Taxonomy backup saved to {backup_filename}")
    except Exception as e:
        print(f"    - Failed to save taxonomy backup: {e}")


def main():
    """Main function to run the taxonomy generation."""
    parser = setup_arg_parser("Generate and populate the food taxonomy.")
    args = parser.parse_args()

    print("--- Starting Improved Food Taxonomy Generation ---")

    backup_file = GENERATED_TAXONOMY_FILE.replace('.json', '_backup.json')
    files_to_remove = [POPULATION_PROGRESS_FILE, backup_file]
    handle_clean_flag(args.clean, files_to_remove)
    
    client = get_openrouter_client()

    print(f"\n1. Reading food classes from source: {SOURCE_META_FILE}")
    try:
        with open(SOURCE_META_FILE, 'r') as f:
            meta_data = json.load(f)
            if "classes" not in meta_data or not isinstance(meta_data["classes"], list):
                raise ValueError("Source file meta.json is missing a 'classes' list.")
            
            all_subcategories = sorted([item['title'] for item in meta_data['classes'] if 'title' in item])

            if not all_subcategories:
                raise ValueError("No classes with a 'title' found in meta.json.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading source file {SOURCE_META_FILE}: {e}")
        return

    # Load progress to handle resumes
    progress = load_progress(POPULATION_PROGRESS_FILE)
    processed_items_map = progress.get("processed_items", {})
    
    # Decide which items to process based on flags
    if args.force:
        print("\n--force flag detected. All items will be re-processed.")
        items_to_process = all_subcategories
    else:
        items_to_process = [item for item in all_subcategories if item not in processed_items_map]
    
    total_items_to_process = len(items_to_process)
    
    if total_items_to_process == 0:
        print("\nAll food items have already been processed. Use --force to re-process.")
        return
        
    print(f"\nFound {total_items_to_process} new or remaining food classes to process.")

    # Initialize taxonomy with previously processed items
    enriched_taxonomy = {parent: [] for parent in PARENT_CATEGORIES}
    for item_name, data in processed_items_map.items():
        parent = data.get("parent_category")
        if parent in enriched_taxonomy:
            enriched_taxonomy[parent].append({
                "name": item_name,
                "display_name": data.get("display_name", item_name),
                "similar_items": data.get("similar_items", [])
            })

    processed_count = 0
    
    print(f"\n--- Processing {total_items_to_process} items in batches of {BATCH_SIZE} ---")
    
    for i in range(0, total_items_to_process, BATCH_SIZE):
        batch = items_to_process[i:i+BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        total_batches = (total_items_to_process + BATCH_SIZE - 1)//BATCH_SIZE
        
        print(f"\n--- Batch [{batch_num}/{total_batches}] ---")
        
        batch_result_map = process_batch(client, batch)

        # Process results
        for original_name, data in batch_result_map.items():
            if original_name not in batch:
                print(f"    - WARNING: Unexpected item '{original_name}' in results")
                continue

            parent = data.get("parent_category")
            similar_items = data.get("similar_items", [])

            if parent not in PARENT_CATEGORIES:
                print(f"    - WARNING: Unknown parent category '{parent}' for '{original_name}'")
                continue

            # Get simplified display name
            display_name = get_simplified_name(client, original_name)
            time.sleep(1.5) # Small delay for the simplification call

            # Create item object
            item_obj = {
                "name": original_name,
                "display_name": display_name,
                "similar_items": sorted(list(set(similar_items)))
            }

            # Add to taxonomy if not duplicate
            if not any(existing_item['name'] == item_obj['name'] for existing_item in enriched_taxonomy.get(parent, [])):
                enriched_taxonomy.setdefault(parent, []).append(item_obj)
                
                # Update progress map with full data including display_name
                progress_data = data.copy()
                progress_data['display_name'] = display_name
                processed_items_map[original_name] = progress_data
                
                processed_count += 1

        print(f"    - Batch complete. Total processed in this run: {processed_count}/{total_items_to_process}")
        
        # Save progress every 5 batches
        if batch_num % 5 == 0:
            save_progress({"processed_items": processed_items_map}, POPULATION_PROGRESS_FILE)
            save_taxonomy_backup(enriched_taxonomy, GENERATED_TAXONOMY_FILE)
        
        time.sleep(RATE_LIMIT_DELAY_SECONDS)

    # Final save of progress and taxonomy
    save_progress({"processed_items": processed_items_map}, POPULATION_PROGRESS_FILE)

    # Final cleanup and sorting
    for parent in enriched_taxonomy:
        enriched_taxonomy[parent] = sorted(enriched_taxonomy[parent], key=lambda x: x['name'])

    final_taxonomy = {k: v for k, v in enriched_taxonomy.items() if v}

    print(f"\n3. Writing final taxonomy to: {GENERATED_TAXONOMY_FILE}")
    try:
        ensure_dir_exists(GENERATED_TAXONOMY_FILE)
        with open(GENERATED_TAXONOMY_FILE, 'w') as f:
            json.dump(final_taxonomy, f, indent=4)
        print("File written successfully")
    except Exception as e:
        print(f"Error writing file: {e}")
        return

    print(f"\n--- Generation Complete ---")
    print(f"Successfully processed: {processed_count}/{total_items_to_process} items")
    if total_items_to_process > 0:
        print(f"Success rate: {processed_count/total_items_to_process*100:.1f}%")
    
    # Summary by category
    print(f"\nItems by category:")
    for category, items in final_taxonomy.items():
        if items:
            print(f"  - {category}: {len(items)} items")

if __name__ == "__main__":
    main() 
