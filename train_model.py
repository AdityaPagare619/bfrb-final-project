#!/usr/bin/env python3
"""
Phase 3.2 • Training Pipeline for BFRB MLP
- Loads augmented features from Phase 2.3
- Applies StandardScaler
- 5-fold stratified CV with class-weighted CrossEntropyLoss
- Mixed precision on CPU with proper autocast syntax
- Checkpoints best model per fold
"""

from pathlib import Path
import yaml, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from model_architecture import BFRBMLP
import joblib

# Paths & config
ROOT      = Path(__file__).resolve().parents[1]
CFG_PATH  = ROOT / "configs" / "training_cfg.yaml"
DATA_FILE = ROOT / "data" / "features" / "bfrb_features_augmented.parquet"
CKPT_DIR  = ROOT / "phase3_modeling" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Load config with proper type casting
cfg = yaml.safe_load(CFG_PATH.read_text())
DEVICE       = torch.device("cpu")
LR           = float(cfg.get("lr", 2.5e-4))
BATCH_SIZE   = int(cfg.get("bs", 128))
EPOCHS       = int(cfg.get("epochs", 40))
WD           = float(cfg.get("weight_decay", 1e-4))
FOLDS        = int(cfg.get("n_folds", 5))

# Load & prepare data
df       = pd.read_parquet(DATA_FILE, engine="pyarrow")
labels   = df["label"].astype("category")
y_all    = labels.cat.codes.values.copy()  # Ensure writable
X_raw    = df.drop(columns="label").values.astype(np.float32).copy()  # Ensure writable

# Scale features
scaler   = StandardScaler().fit(X_raw)
X_scaled = scaler.transform(X_raw)
joblib.dump(scaler, ROOT / "phase3_modeling" / "scaler.pkl")

# Create tensors from writable arrays
X_tensor = torch.from_numpy(X_scaled)
y_tensor = torch.from_numpy(y_all).long()
dataset  = TensorDataset(X_tensor, y_tensor)

n_features = X_scaled.shape[1]
n_classes  = len(labels.cat.categories)

# Class weights
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_all), y=y_all
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Training/eval epoch with FIXED autocast syntax
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, trues = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # FIXED: Specify device_type for autocast
            with autocast(device_type="cpu"):
                logits = model(xb)
                loss   = criterion(logits, yb)
            
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * xb.size(0)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    
    f1 = f1_score(trues, preds, average="macro")
    return total_loss / len(loader.sampler), f1

# Cross-validation
print("=== BFRB Model Training ===")
print(f"Features: {n_features}, Classes: {n_classes}")
print(f"Dataset size: {len(dataset)}")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_all), 1):
    print(f"\n--- Fold {fold}/{FOLDS} ---")
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_loader   = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(val_idx))

    model     = BFRBMLP(n_features, n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scaler    = GradScaler(device="cpu")  # Specify device for GradScaler

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_f1 = run_epoch(model, val_loader, criterion)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train F1: {tr_f1:.3f} | Val F1: {va_f1:.3f}")
        
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), CKPT_DIR / f"bfrb_fold{fold}.pt")

    print(f"★ Fold {fold} Best Val F1: {best_f1:.4f}")
    fold_scores.append(best_f1)

print(f"\n=== Training Complete ===")
print(f"CV Macro-F1 scores: {fold_scores}")
print(f"Mean Macro-F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"Checkpoints saved to: {CKPT_DIR}")
