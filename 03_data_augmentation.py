#!/usr/bin/env python3
"""
Phase 2.3 • Ethical Data Augmentation for BFRB Detection
Version: 3.2 (2025-08-01)

Reads bfrb_features_matrix.parquet and feature_metadata.parquet,
applies balanced, label-aware augmentation to the feature vectors,
and outputs bfrb_features_augmented.parquet for Phase 3 training.

Augmentation techniques:
 - Gaussian noise injection
 - Random feature scaling
 - Feature dropout
 - Temporal jitter
 - Synthetic interpolation (mixup)
"""

import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Paths
ROOT           = Path(__file__).resolve().parents[1] / "data" / "features"
INPUT_FEATURES = ROOT / "bfrb_features_matrix.parquet"
INPUT_META     = ROOT / "feature_metadata.parquet"
OUTPUT_PATH    = ROOT / "bfrb_features_augmented.parquet"

class BFRBAugmenter:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def gaussian_noise(self, X, scale: float=0.02):
        noise = self.rng.normal(0, scale, X.shape)
        return X + noise * X.std(axis=0, keepdims=True)

    def random_scaling(self, X, low: float=0.95, high: float=1.05):
        factors = self.rng.uniform(low, high, size=(1, X.shape[1]))
        return X * factors

    def feature_dropout(self, X, rate: float=0.05):
        mask = self.rng.rand(*X.shape) > rate
        return X * mask

    def temporal_jitter(self, X, strength: float=0.02):
        Xj = X.copy()
        n_feat = X.shape[1]
        n_jit = max(1, int(n_feat * strength))
        for i in range(X.shape[0]):
            idx = self.rng.choice(n_feat, n_jit, replace=False)
            Xj[i, idx] = self.rng.permutation(Xj[i, idx])
        return Xj

    def mixup(self, X, y, alpha: float=0.3):
        lam = self.rng.beta(alpha, alpha)
        idx = self.rng.randint(0, len(X))
        X2, y2 = X[idx], y[idx]
        return lam*X + (1-lam)*X2, y

def compute_targets(counter: Counter, multiplier: float):
    max_count = max(counter.values())
    total_target = int(sum(counter.values()) * multiplier)
    avg_target_per_class = total_target // len(counter)
    return {label: max(avg_target_per_class - count, 0)
            for label, count in counter.items()}

def main():
    parser = argparse.ArgumentParser(description="03_data_augmentation")
    parser.add_argument("--multiplier", type=float, default=4.0,
                        help="Total dataset size multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load features and metadata
    if not INPUT_FEATURES.exists():
        print("❌ Missing feature matrix; run Phase 2.2 first."); sys.exit(1)
    df = pd.read_parquet(INPUT_FEATURES, engine="pyarrow")
    y = df["label"].values
    X = df.drop(columns="label").to_numpy(dtype=np.float32)

    # Compute augmentation targets
    class_counts = Counter(y)
    targets = compute_targets(class_counts, args.multiplier)
    print("📊 Original class distribution:", dict(class_counts))
    print("🎯 Augmentation targets per class:", targets)

    augmenter = BFRBAugmenter(seed=args.seed)
    X_list, y_list = [X], [y]

    # Augment each class
    for label, count in targets.items():
        if count <= 0:
            continue
        mask = (y == label)
        Xc, yc = X[mask], y[mask]
        n = len(Xc)
        if n == 0:
            continue
        print(f"⚙️ Augmenting class '{label}': need {count} samples from {n} originals")
        generated = 0
        while generated < count:
            # Choose random sample
            idx = augmenter.rng.randint(0, n)
            x0, y0 = Xc[idx:idx+1], np.array([label])
            method = augmenter.rng.choice(
                ["gaussian", "scaling", "dropout", "jitter", "mixup"],
                p=[0.3,0.25,0.2,0.15,0.1]
            )
            if method == "gaussian":
                xa = augmenter.gaussian_noise(x0)
            elif method == "scaling":
                xa = augmenter.random_scaling(x0)
            elif method == "dropout":
                xa = augmenter.feature_dropout(x0)
            elif method == "jitter":
                xa = augmenter.temporal_jitter(x0)
            else:  # mixup
                xa, _ = augmenter.mixup(x0.flatten(), y)
                xa = xa.reshape(1, -1)
            X_list.append(xa)
            y_list.append(y0)
            generated += 1

    # Combine and shuffle
    X_aug = np.vstack(X_list)
    y_aug = np.concatenate(y_list)
    perm = np.arange(len(X_aug))
    augmenter.rng.shuffle(perm)
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    # Save augmented DataFrame
    df_aug = pd.DataFrame(X_aug, columns=df.columns.drop("label"))
    df_aug["label"] = y_aug
    df_aug.to_parquet(OUTPUT_PATH, engine="pyarrow",
                      compression="zstd", version="1.0",
                      use_deprecated_int96_timestamps=False)
    print(f"✅ Augmented dataset saved: {OUTPUT_PATH.name}")
    print(f"📊 New class distribution: {Counter(y_aug)}")

if __name__ == "__main__":
    main()
