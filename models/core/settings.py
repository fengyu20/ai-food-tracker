"""
Centralized configuration for file paths used across the project.
This ensures that all modules refer to the same file locations.
"""

# --- Source Data ---
SOURCE_META_FILE = 'content/extracted_food_recognition/meta.json'

# --- Generated Taxonomy Files ---
# Output from the initial population script
GENERATED_TAXONOMY_FILE = 'config/categories_mapping_generated.json'

# Output from the refinement script (the final, canonical taxonomy)
REFINED_TAXONOMY_FILE = 'config/categories_mapping_refined.json'

# --- Progress & Report Files ---
POPULATION_PROGRESS_FILE = 'config/population_progress.json'
REFINEMENT_PROGRESS_FILE = 'config/refinement_progress.json'
REFINEMENT_REPORT_FILE = REFINED_TAXONOMY_FILE.replace('.json', '_refinement_report.txt') 