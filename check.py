import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# === Load data ===
data_path = "./aligned_features/svm_add_data_features_labels.npz"
data = np.load(data_path)
X, y = data["X"], data["y"]

print(f"Loaded: X={X.shape}, y={y.shape}")
unique, counts = np.unique(y, return_counts=True)
print("Initial class distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,} samples")

# # === Filter out classes with fewer than 200 samples ===
# unique, counts = np.unique(y, return_counts=True)
# valid_classes = unique[counts >= 200]
# mask_valid = np.isin(y, valid_classes)
# X = X[mask_valid]
# y = y[mask_valid]

# print(f"After filtering: {len(valid_classes)} valid classes, {X.shape[0]:,} samples remain")

# # === Small validation subset (10%) ===
# Xv, _, yv, _ = train_test_split(
#     X, y,
#     test_size=0.9,
#     stratify=y,
#     random_state=42
# )
# print(f"Validation subset: {Xv.shape[0]} samples")

# # === Quick linear SVM check ===
# pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm", LinearSVC(class_weight="balanced", max_iter=2000, dual=False))
# ])
# pipe.fit(Xv, yv)
# print("\nSVM trained for quick check.")
# print(classification_report(yv, pipe.predict(Xv)))

# # === Feature importance (coef_) ===
# svm = pipe.named_steps["svm"]
# if hasattr(svm, "coef_"):
#     importance = np.mean(np.abs(svm.coef_), axis=0)
#     plt.figure(figsize=(10, 4))
#     plt.bar(range(len(importance)), importance)
#     plt.title("Feature Importance (|coef_| from LinearSVC)")
#     plt.xlabel("Feature index")
#     plt.ylabel("Relative importance")
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No coef_ attribute available for this model type.")
