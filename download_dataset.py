import os
import tarfile
import shutil
import argparse
from tqdm import tqdm


def download_and_extract(dataset_type="sample", extract_dir="/content/extracted_food_recognition", force=False):

    urls = {
        "full": "https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzk1MF9Gb29kIFJlY29nbml0aW9uIDIwMjIvZm9vZC1yZWNvZ25pdGlvbi0yMDIyLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogIlpqZisyZURmaEoyZkhWNGRiTHBPWkEzN0NodWhlb28wNlZlQXpQQkdBc1U9In0=",
        "sample": "https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/Y/I/Yu/V12n4DX73dwmLY7sz7Bl83qOdACZBbaB9ctVmlEUPBx1qqaqLqnHsLubJQWGAKu3vrqBPty6hRBOrUaXzgPe1jMk7bI1MVcWCNON9vbLeZdPKuRV9Psis3STGSh7.tar"
    }

    if dataset_type not in urls:
        raise ValueError("dataset_type must be either 'full' or 'sample'")

    url = urls[dataset_type]
    tar_path = f"/content/{dataset_type}_food_dataset.tar"

    # If force = False and folder exists, skip download & extraction
    if os.path.exists(extract_dir) and not force:
        print(f"{extract_dir} already exists. Skipping download & extraction.")
        return

    # If force = True and directory exists, remove it before download & extraction
    if os.path.exists(extract_dir) and force:
        print(f"Removing existing directory {extract_dir}...")
        shutil.rmtree(extract_dir)

    print(f"Downloading [{dataset_type}] dataset...")
    os.system(f'wget -q --show-progress -O "{tar_path}" "{url}"')

    print(f"Extracting to {extract_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        for member in tqdm(tar.getmembers(), desc="Extracting"):
            tar.extract(member, path=extract_dir)

    print(f"Dataset extracted to {extract_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract food recognition dataset")
    parser.add_argument("--type", choices=["full", "sample"], default="sample", help="Which dataset to download")
    parser.add_argument("--dir", default="/content/extracted_food_recognition", help="Where to extract dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite existing data")
    args = parser.parse_args()

    download_and_extract(dataset_type=args.type, extract_dir=args.dir, force=args.force)