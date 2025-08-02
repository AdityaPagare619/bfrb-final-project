#!/usr/bin/env python3
"""
Phase 3.2 • Training Pipeline for BFRB MLP
- Loads augmented features from Phase 2.3
- Applies StandardScaler
- 5-fold stratified CV with class-weighted CrossEntropyLoss
- Mixed precision on CPU
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

# Paths & config
ROOT      = Path(__file__).resolve().parents[1]
CFG       = yaml.safe_load((ROOT/"configs"/"training_cfg.yaml").read_text())
DATA_FILE = ROOT/"data"/"features"/"bfrb_features_augmented.parquet"
CKPT_DIR  = ROOT/"phase3_modeling"/"checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Hyperparams
DEVICE = torch.device("cpu")
LR = float(CFG.get("lr", 2.5e-4))
BATCH_SIZE = int(CFG.get("bs", 128))
EPOCHS = int(CFG.get("epochs", 40))
WD = float(CFG.get("weight_decay", 1e-4))
FOLDS = int(CFG.get("n_folds", 5))

# Load & scale data
df       = pd.read_parquet(DATA_FILE, engine="pyarrow")
labels   = df["label"].astype("category")
y_all    = labels.cat.codes.values
X_raw    = df.drop(columns="label").values.astype(np.float32)

scaler   = StandardScaler().fit(X_raw)
X_scaled = scaler.transform(X_raw)
import joblib; joblib.dump(scaler, ROOT/"phase3_modeling"/"scaler.pkl")

# Tensors & dataset
X_tensor = torch.from_numpy(X_scaled)
y_tensor = torch.from_numpy(y_all.copy()).long()
dataset  = TensorDataset(X_tensor, y_tensor)

n_features = X_scaled.shape[1]
n_classes  = len(labels.cat.categories)

# Class weights
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_all), y=y_all
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Training/eval epoch
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, trues = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast():
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
    return total_loss/len(loader.sampler), f1

# Cross-validation
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_all), 1):
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_loader   = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(val_idx))

    model     = BFRBMLP(n_features, n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scaler    = GradScaler()

    best_f1 = 0.0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_f1 = run_epoch(model, val_loader,   criterion)
        print(f"[Fold {fold}] Epoch {epoch}/{EPOCHS} | "
              f"Train F1 {tr_f1:.3f} | Val F1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), CKPT_DIR/f"bfrb_fold{fold}.pt")

    print(f"★ Fold {fold} best Val F1: {best_f1:.4f}\n")
    fold_scores.append(best_f1)

print("CV Macro-F1 scores:", fold_scores)
print(f"Mean Macro-F1: {np.mean(fold_scores):.4f}")
