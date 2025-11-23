import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download, utils as hf_utils


logger = logging.getLogger(__name__)

def download_dataset(
    repo_id: str, 
    output_dir: Optional[Path] = None, 
    force_download: bool = False
) -> Path:
    """
    Downloads a dataset from Hugging Face Hub.

    Args:
        repo_id: The HF repository ID (e.g., 'NoeFontana/cs336').
        output_dir: The local directory to save files. Defaults to ~/datasets/<name>.
        force_download: If True, forces redownloading even if files exist.

    Returns:
        Path: The absolute path to the downloaded dataset.
    """
    # Determine output directory
    if output_dir is None:
        dataset_name = repo_id.split("/")[-1]
        destination = Path(f"~/datasets/{dataset_name}").expanduser()
    else:
        destination = output_dir.expanduser()

    logger.info(f"⬇️  Starting download for '{repo_id}' to '{destination}'...")

    try:
        # snapshot_download handles caching and file retrieval
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=destination,
            force_download=force_download,
        )

        logger.info(f"✅ Success! Files are available at: {local_path}")
        
        # Quick content summary
        file_count = len(list(Path(local_path).rglob('*')))
        logger.info(f"📂 Total files present: {file_count}")
        
        return Path(local_path)

    except hf_utils.RepositoryNotFoundError:
        logger.error(f"❌ Repository '{repo_id}' not found. Check the spelling or visibility.")
        sys.exit(1)
    except hf_utils.GatedRepoError:
        logger.error(f"❌ Repository '{repo_id}' is gated. Ensure HF_TOKEN is set and you have accepted the license.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face Hub.")
    
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="NoeFontana/cs336", 
        help="The Hugging Face repository ID."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=None, 
        help="Custom output directory. Defaults to ~/datasets/<repo_name>."
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force redownload of files."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    download_dataset(args.repo_id, args.output_dir, args.force)