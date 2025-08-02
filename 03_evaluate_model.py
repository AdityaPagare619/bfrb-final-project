#!/usr/bin/env python3
"""
Phase 3.3 • Model Evaluation & Export for BFRB Detection

- Loads best fold checkpoint from Phase 3.2
- Evaluates on the entire augmented dataset
- Saves confusion matrix (CSV + PNG), classification report (TXT), ROC-AUC per class (CSV)
- Exports model to TorchScript and ONNX for Phase 4 deployment
"""

from pathlib import Path
import yaml, numpy as np, pandas as pd, torch, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from model_architecture import BFRBMLP

# Paths
ROOT       = Path(__file__).resolve().parents[1]
CFG_PATH   = ROOT / "configs" / "training_cfg.yaml"
DATA_PATH  = ROOT / "data" / "features" / "bfrb_features_augmented.parquet"
SCALER_PATH= ROOT / "phase3_modeling" / "scaler.pkl"
CKPT_DIR   = ROOT / "phase3_modeling" / "checkpoints"
EXPORT_DIR = ROOT / "phase3_modeling" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load config
cfg    = yaml.safe_load(CFG_PATH.read_text())
DEVICE = torch.device("cpu")

# Load data
df      = pd.read_parquet(DATA_PATH, engine="pyarrow")
labels  = df["label"].astype("category")
y_true  = labels.cat.codes.values
X_raw   = df.drop(columns="label").values.astype(np.float32)

# Scale features
scaler  = joblib.load(SCALER_PATH)
X_scaled= scaler.transform(X_raw)
X_tensor= torch.from_numpy(X_scaled).to(DEVICE)

# Determine best checkpoint (highest fold number)
ckpts = sorted(CKPT_DIR.glob("bfrb_fold*.pt"))
best_ckpt = ckpts[-1]
print("→ Loading checkpoint:", best_ckpt.name)

# Load model
model = BFRBMLP(input_dim=X_scaled.shape[1],
               num_classes=len(labels.cat.categories)).to(DEVICE)
model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
model.eval()

# Inference
with torch.no_grad():
    logits = model(X_tensor)
    y_pred = logits.argmax(dim=1).cpu().numpy()
    probas = torch.softmax(logits, dim=1).cpu().numpy()

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
classes = labels.cat.categories.tolist()
n_classes = len(classes)
cm_df = pd.DataFrame(cm, index=labels.cat.categories, columns=labels.cat.categories)
cm_df.to_csv(EXPORT_DIR / "confusion_matrix.csv")

plt.figure(figsize=(8,8))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.xticks(range(n_classes), classes, rotation=90)
plt.yticks(range(n_classes), classes)
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig(EXPORT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# 2. Classification Report
report = classification_report(y_true, y_pred,
                               target_names=labels.cat.categories,
                               zero_division=0)
with open(EXPORT_DIR / "classification_report.txt", "w") as f:
    f.write(report)
print(report)

# 3. ROC Curves (One-vs-Rest)
fpr, tpr, roc_auc = {}, {}, {}
y_onehot = np.eye(len(labels))[y_true]
for i, cls in enumerate(labels.cat.categories):
    fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i, cls in enumerate(labels.cat.categories):
    plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.2f})")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(EXPORT_DIR / "roc_curves.png", dpi=200)
plt.close()

# 4. Export TorchScript
example = torch.randn(1, X_scaled.shape[1]).to(DEVICE)
ts_mod  = torch.jit.trace(model, example)
ts_mod.save(EXPORT_DIR / "bfrb_model_cpu.pt")

# 5. Export ONNX
onnx_path = EXPORT_DIR / "bfrb_model_cpu.onnx"
torch.onnx.export(
    model, example, onnx_path,
    input_names=["input"], output_names=["logits"],
    opset_version=13, do_constant_folding=True
)

print("Exports completed. Files saved to:", EXPORT_DIR)
