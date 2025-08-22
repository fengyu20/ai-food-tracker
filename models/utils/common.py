import argparse
import os
import json
import re
from typing import List, Dict, Set


def normalize_item_name(name: str) -> str:
    """Normalizes an item name for consistent comparison."""
    # Convert to lowercase
    name = name.lower()
    # Remove content in parentheses
    name = re.sub(r'\s*\([^)]*\)', '', name)
    # Replace non-alphanumeric characters (except spaces) with a space
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    # Replace multiple spaces with a single hyphen and strip leading/trailing hyphens
    name = re.sub(r'\s+', '-', name.strip())
    return name

# Unified tokens used across lemma and semantic normalization
PREP_TOKENS: Set[str] = {
  "raw","steamed","stewed","grilled","fried","roasted","baked",
  "boiled","sauteed","poached","creamed","cream","sauce","dip",
  "with","without","addition","of","fat","salt","and","in","made","from",
  "cooked","fresh","dried","frozen","smoked","sliced","slice","chopped","toasted",
  "seed","seeds","ball","bun","roll","batter","coated","mix","mixed","style",
  "canned","preserved","natural","unspiced","half","whole","average",
  "curd","curds","philadelphia",  
  "frying","puff","pastry",
  "smoked","boiled","filling"
}

def _singularize(token: str) -> str:
    if not token:
        return token
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def naturalize(name: str) -> str:
    """Convert hyphenated taxonomy names to natural language, dropping prep tokens and singularizing nouns."""
    n = normalize_item_name(name or "")
    if not n:
        return n
    toks = [t for t in n.split('-') if t and t not in PREP_TOKENS]
    toks = [_singularize(t) for t in toks]
    return " ".join([t for t in toks if t])


def food_lemma(name: str) -> str:
    """Extracts the base food concept by removing preparation/modifier tokens and singularizing.

    The result is a compact lemma of 1-2 salient tokens to avoid over-generalization.
    """
    norm = normalize_item_name(name or "")
    if not norm:
        return norm

    tokens = [t for t in norm.split('-') if t and t not in PREP_TOKENS]
    tokens = [_singularize(t) for t in tokens]

    core_tokens: List[str] = []
    for token in tokens:
        # Drop generic qualifiers and nutrition adjectives
        if token in {"high", "low", "reduced", "protein", "fiber", "plant", "legume"}:
            continue
        if token in {"made", "from", "addition", "without", "with", "and", "in"}:
            continue
        core_tokens.append(token)
        if len(core_tokens) >= 2:
            break

    if core_tokens:
        return '-'.join(core_tokens)
    # Fallback to first remaining token or the normalized name
    return tokens[0] if tokens else norm


def setup_arg_parser(description: str) -> argparse.ArgumentParser:
    """Sets up a standard argument parser with --force and --clean flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all items, even if they exist in the progress file."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Start from scratch, deleting existing progress and backup files."
    )
    return parser


def handle_clean_flag(clean_flag: bool, files_to_remove: List[str]):
    """Removes specified files if the --clean flag is set."""
    if clean_flag:
        print("\n--clean flag detected. Removing existing progress and backup files.")
        for f_path in files_to_remove:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"    - Removed {f_path}")
                except OSError as e:
                    print(f"    - Error removing file {f_path}: {e}")


def ensure_dir_exists(filepath: str):
    """Ensure the directory for a given file path exists before writing."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"    - Created directory: {directory}")


def load_progress(progress_file: str) -> Dict:
    """Load existing progress to resume if interrupted."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"    - Error loading progress from {progress_file}: Invalid JSON.")
            return {}
    return {}


def save_progress(progress_data: Dict, progress_file: str):
    """Save the dictionary of processed items to the progress file."""
    try:
        ensure_dir_exists(progress_file)
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"    - Failed to save progress: {e}")
