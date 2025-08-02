#!/usr/bin/env python3
"""
Phase 2.2 • Robust Feature Extraction for BFRB Detection
Version: 3.3 (2025-08-01)

1. Loads plain Parquet sensor data from Phase 2.1.
2. Generates sliding windows, including a final partial window.
3. Extracts time-domain, frequency-domain, physiological, and cross-modal features.
4. Skips windows smaller than 50% of WINDOW_SIZE.
5. Logs per-window errors without halting.
6. Writes feature matrix and metadata as plain Parquet (Parquet v1.0).
"""

import argparse, gc, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Parameters
WINDOW_SIZE     = 500
STEP_SIZE       = 250
MIN_WINDOW_SIZE = WINDOW_SIZE // 2
IMU_FS          = 100
DIST_FS         = 20
HR_FS           = 25

# Paths
ROOT         = Path(__file__).resolve().parents[1] / "data"
PARQUET_DIR  = ROOT / "parquet"
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def sliding_windows(total_len: int):
    windows = []
    start = 0
    while start + WINDOW_SIZE <= total_len:
        windows.append((start, start + WINDOW_SIZE))
        start += STEP_SIZE
    # final partial window if large enough
    if start < total_len and total_len - start >= MIN_WINDOW_SIZE:
        windows.append((start, total_len))
    return windows

def stat_features(x, prefix):
    feats = {}
    if x.size == 0 or np.allclose(x, 0):
        return feats
    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_std"]  = float(np.std(x))
    feats[f"{prefix}_min"]  = float(np.min(x))
    feats[f"{prefix}_max"]  = float(np.max(x))
    feats[f"{prefix}_ptp"]  = float(np.ptp(x))
    feats[f"{prefix}_rms"]  = float(np.sqrt(np.mean(x**2)))
    feats[f"{prefix}_skew"] = float(stats.skew(x)) if x.size>2 else 0.0
    feats[f"{prefix}_kurt"] = float(stats.kurtosis(x)) if x.size>3 else 0.0
    for p in (10,25,75,90):
        feats[f"{prefix}_p{p}"] = float(np.percentile(x, p))
    feats[f"{prefix}_zero_crossings"] = float(np.count_nonzero(np.diff(np.sign(x))))
    return feats

def spectral_features(x, fs, prefix):
    feats = {}
    if x.size < 8: return feats
    freqs = np.fft.rfftfreq(x.size, 1/fs)
    mags  = np.abs(np.fft.rfft(x))
    total = mags.sum()
    if total == 0: return feats
    centroid = float((freqs*mags).sum()/total)
    spread   = float(np.sqrt(((freqs-centroid)**2*mags).sum()/total))
    rolloff_idx = np.where(np.cumsum(mags) >= 0.85*total)[0]
    rolloff = float(freqs[rolloff_idx[0]]) if rolloff_idx.size else 0.0
    dominant = float(freqs[np.argmax(mags)])
    feats[f"{prefix}_centroid"] = centroid
    feats[f"{prefix}_spread"]   = spread
    feats[f"{prefix}_rolloff"]  = rolloff
    feats[f"{prefix}_dominant"] = dominant
    return feats

class FeatureExtractor:
    def extract_imu(self, seg):
        feats = {}
        for col in ('acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'):
            if col in seg:
                arr = seg[col].to_numpy(dtype=np.float32)
                feats |= stat_features(arr, col)
                feats |= spectral_features(arr, IMU_FS, col)
        if all(c in seg for c in ('acc_x','acc_y','acc_z')):
            mag = np.linalg.norm(seg[['acc_x','acc_y','acc_z']].to_numpy(), axis=1)
            feats |= stat_features(mag, "acc_mag")
            feats |= spectral_features(mag, IMU_FS, "acc_mag")
            try:
                peaks, _ = signal.find_peaks(mag, prominence=np.std(mag)*0.5)
                feats["motion_peak_count"] = float(len(peaks))
                feats["motion_peak_rate"]  = float(len(peaks)/(len(mag)/IMU_FS))
            except:
                feats["motion_peak_count"] = 0.0
                feats["motion_peak_rate"]  = 0.0
        return feats

    def extract_distance(self, seg):
        feats = {}
        if 'distance' in seg:
            arr = seg['distance'].to_numpy(dtype=np.float32)
            feats |= stat_features(arr, "dist")
            feats |= spectral_features(arr, DIST_FS, "dist")
            diff = np.diff(arr)
            feats["approach_rate"] = float(np.mean(diff<0))
            feats["withdraw_rate"] = float(np.mean(diff>0))
        return feats

    def extract_physio(self, seg):
        feats = {}
        if 'heart_rate' in seg:
            arr = seg['heart_rate'].to_numpy(dtype=np.float32)
            feats |= stat_features(arr, "hr")
        if 'spo2' in seg:
            arr = seg['spo2'].to_numpy(dtype=np.float32)
            feats |= stat_features(arr, "spo2")
        return feats

    def extract_cross(self, seg):
        feats = {}
        if all(c in seg for c in ('acc_x','acc_y','acc_z','distance')):
            acc = np.linalg.norm(seg[['acc_x','acc_y','acc_z']].to_numpy(),axis=1)
            dist = seg['distance'].to_numpy(dtype=np.float32)
            if acc.size==dist.size and dist.size>3:
                corr = np.corrcoef(acc, dist)[0,1]
                feats["motion_dist_corr"] = float(corr) if not np.isnan(corr) else 0.0
        return feats

    def extract_window(self, seg):
        feats = {}
        feats |= self.extract_imu(seg)
        feats |= self.extract_distance(seg)
        feats |= self.extract_physio(seg)
        feats |= self.extract_cross(seg)
        return feats

def main():
    print("=== BFRB Advanced Feature Extraction v3.3 ===")
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        print("❌ No Parquet files. Run Phase 2.1 first."); sys.exit(1)
    print(f"📁 Found {len(files)} Parquet files")
    print(f"🪟 Window: {WINDOW_SIZE}, Step: {STEP_SIZE}, Min: {MIN_WINDOW_SIZE}")

    extractor = FeatureExtractor()
    all_feats, all_lbls, all_meta = [], [], []

    for pq in files:
        print(f"\n🔄 Processing {pq.name}")
        df = pd.read_parquet(pq, engine="pyarrow")
        wins = sliding_windows(len(df))
        print(f"  - {len(wins)} windows")

        for idx,(s,e) in enumerate(wins):
            seg = df.iloc[s:e]
            if len(seg) < MIN_WINDOW_SIZE:
                continue
            try:
                feats = extractor.extract_window(seg)
                if not feats:
                    continue
                label = seg.get("label", seg.get("gesture", pd.Series(["unknown"]))).iloc[0]
                all_feats.append(feats)
                all_lbls.append(label)
                all_meta.append({
                    "sequence_id": f"{pq.stem}_{idx}",
                    "file": pq.name, "start": s, "end": e,
                    "label": label
                })
            except Exception as ex:
                print(f"  ⚠️ Window {idx} error: {ex}")
        del df; gc.collect()

    if not all_feats:
        print("❌ No features extracted"); sys.exit(1)

    feat_df = pd.DataFrame(all_feats).fillna(0).astype(np.float32)
    feat_df = feat_df.loc[:, feat_df.var() > 1e-8]
    feat_df["label"] = all_lbls
    meta_df = pd.DataFrame(all_meta)

    feat_df.to_parquet(
        FEATURES_DIR/"bfrb_features_matrix.parquet",
        engine="pyarrow", compression="zstd",
        version="1.0", use_deprecated_int96_timestamps=False
    )
    meta_df.to_parquet(
        FEATURES_DIR/"feature_metadata.parquet",
        engine="pyarrow", compression="zstd",
        version="1.0", use_deprecated_int96_timestamps=False
    )

    print(f"\n✅ Saved features: {feat_df.shape}")
    print(f"✅ Saved metadata: {meta_df.shape}")

if __name__ == "__main__":
    main()
