"""
Dataset manager for the Emotion-Aware Sarcasm Detection System.

Downloads and prepares the News Headlines Dataset for Sarcasm Detection (Misra, 2019).
Creates reproducible train / validation / test splits (FR-5.1, FR-5.5).
"""

import json
import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetManager:
    """Handles dataset download, loading and splitting."""

    def __init__(self, config, data_dir=None):
        self.config = config
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__))

    def download(self):
        """Download the sarcasm dataset if not already present.

        Tries the primary URL first, then the fallback URL.
        """
        filepath = os.path.join(self.data_dir, "raw_data.json")
        if os.path.exists(filepath):
            print(f"[Dataset] Raw data already exists at {filepath}")
            return filepath

        urls = [self.config["dataset_url"]]
        if "dataset_url_fallback" in self.config:
            urls.append(self.config["dataset_url_fallback"])

        for url in urls:
            try:
                print(f"[Dataset] Downloading dataset from {url} ...")
                response = requests.get(url, timeout=120)
                response.raise_for_status()

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(response.text)

                print(f"[Dataset] Saved to {filepath}")
                return filepath

            except Exception as e:
                print(f"[Dataset] Failed to download from {url}: {e}")
                continue

        raise RuntimeError(
            "Could not download the dataset from any source. "
            "Please download manually and place as data/raw_data.json"
        )

    def load_raw(self, filepath=None):
        """Load raw dataset into a DataFrame.

        Handles both JSON-lines format and JSON array format.
        """
        if filepath is None:
            filepath = self.download()

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Detect format: JSON array vs JSON-lines
        if content.startswith("["):
            # JSON array format (e.g., TensorFlow-hosted sarcasm.json)
            records = json.loads(content)
        else:
            # JSON-lines format (one JSON object per line)
            records = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        df = pd.DataFrame(records)

        # Normalise column names
        if "headline" in df.columns:
            df = df.rename(columns={"headline": "text"})
        if "is_sarcastic" in df.columns:
            df = df.rename(columns={"is_sarcastic": "label"})

        df = df[["text", "label"]].copy()
        df["label"] = df["label"].astype(int)

        # Remove duplicates and empty texts
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

        print(f"[Dataset] Loaded {len(df)} records")
        print(f"[Dataset] Class distribution: {df['label'].value_counts().to_dict()}")
        return df

    def split(self, df=None, max_samples=None):
        """Create reproducible train / validation / test splits.

        Splits are created before model selection (Design constraint in SRS).
        Stratified splitting preserves class balance.
        """
        if df is None:
            df = self.load_raw()

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.config["random_seed"])
            df = df.reset_index(drop=True)
            print(f"[Dataset] Subsampled to {len(df)} records (quick mode)")

        seed = self.config["random_seed"]
        test_size = self.config["test_split"]
        val_size = self.config["val_split"]

        # First split: separate test set
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["label"]
        )

        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=seed, stratify=train_val["label"]
        )

        # Reset indices
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # Save splits
        train.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.data_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)

        print(f"[Dataset] Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def load_splits(self):
        """Load previously saved splits from CSV files."""
        train = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        val = pd.read_csv(os.path.join(self.data_dir, "val.csv"))
        test = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        return train, val, test

    def get_dataset_info(self):
        """Return dataset metadata for reporting."""
        info = {
            "name": "News Headlines Dataset for Sarcasm Detection v2",
            "source": "Rishabh Misra (2019)",
            "url": self.config["dataset_url"],
            "licence": "CC BY-SA 4.0 (academic use)",
            "label_mapping": {"0": "non-sarcastic", "1": "sarcastic"},
            "random_seed": self.config["random_seed"],
            "test_split_ratio": self.config["test_split"],
            "val_split_ratio": self.config["val_split"],
        }

        # Add split sizes if available
        for split_name in ["train", "val", "test"]:
            path = os.path.join(self.data_dir, f"{split_name}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                info[f"{split_name}_size"] = len(df)
                info[f"{split_name}_class_distribution"] = df["label"].value_counts().to_dict()

        return info
