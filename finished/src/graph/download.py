"""Dataset download and validation for Elliptic and IBM AML datasets."""

import os
import subprocess
from pathlib import Path

from loguru import logger


ELLIPTIC_EXPECTED_FILES = [
    "elliptic_txs_classes.csv",
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
]

IBM_AML_EXPECTED_FILES = [
    "HI-Small_Trans.csv",
]

ELLIPTIC_KAGGLE_DATASET = "ellipticco/elliptic-data-set"
IBM_AML_KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


def download_elliptic(raw_dir: str = "data/raw") -> Path:
    """
    Download Elliptic Bitcoin dataset from Kaggle.

    Requires the Kaggle CLI (`pip install kaggle`) and a configured
    ~/.kaggle/kaggle.json API token.

    If files already exist, skips download.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if validate_dataset("elliptic", raw_dir):
        logger.info("Elliptic dataset already present, skipping download.")
        return raw_path

    logger.info("Downloading Elliptic dataset from Kaggle...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", ELLIPTIC_KAGGLE_DATASET,
                "-p", str(raw_path),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle` "
            "and configure ~/.kaggle/kaggle.json"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e.stderr}")

    if not validate_dataset("elliptic", raw_dir):
        raise RuntimeError(
            f"Download completed but expected files not found in {raw_path}. "
            "You may need to manually extract or reorganize the files."
        )

    logger.info("Elliptic dataset downloaded successfully.")
    return raw_path


def download_ibm_aml(raw_dir: str = "data/raw") -> Path:
    """
    Download IBM Synthetic AML dataset (HI-Small variant) from Kaggle.

    Requires the Kaggle CLI and configured API token.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if validate_dataset("ibm_aml", raw_dir):
        logger.info("IBM AML dataset already present, skipping download.")
        return raw_path

    logger.info("Downloading IBM AML dataset from Kaggle...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", IBM_AML_KAGGLE_DATASET,
                "-p", str(raw_path),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle` "
            "and configure ~/.kaggle/kaggle.json"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e.stderr}")

    if not validate_dataset("ibm_aml", raw_dir):
        raise RuntimeError(
            f"Download completed but expected files not found in {raw_path}. "
            "You may need to manually extract or reorganize the files."
        )

    logger.info("IBM AML dataset downloaded successfully.")
    return raw_path


def validate_dataset(dataset_name: str, raw_dir: str = "data/raw") -> bool:
    """Check that expected files exist for the given dataset."""
    raw_path = Path(raw_dir)

    if dataset_name == "elliptic":
        expected = ELLIPTIC_EXPECTED_FILES
    elif dataset_name == "ibm_aml":
        expected = IBM_AML_EXPECTED_FILES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Check both flat and nested structures (Kaggle sometimes nests)
    for filename in expected:
        flat = raw_path / filename
        if flat.exists():
            continue
        # Search one level deep
        found = list(raw_path.rglob(filename))
        if not found:
            logger.debug(f"Missing file: {filename}")
            return False

    return True


def find_file(filename: str, raw_dir: str = "data/raw") -> Path:
    """Find a dataset file, checking both flat and nested directory structures."""
    raw_path = Path(raw_dir)

    flat = raw_path / filename
    if flat.exists():
        return flat

    found = list(raw_path.rglob(filename))
    if found:
        return found[0]

    raise FileNotFoundError(f"Cannot find {filename} in {raw_dir}")
