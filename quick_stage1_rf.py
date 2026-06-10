# quick_stage1_rf.py
import numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix
data = np.load("./aligned_features/svm_features_labels.npz")
X, y = data['X'], data['y']
mask = (y!=0)&(y!=32767)
X, y = X[mask], y[mask]
# keep LU codes with at least 200 pix and sample per-LU as before (reuse your filter_and_sample function)
# for brevity here we'll map directly to superclasses and sample balanced by super-class
# Map to super
econ_set = set([2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420])
water_set = set([4101,4102,4103,4201,4202,4203])
y_super = np.array([1 if v in econ_set else (2 if v in water_set else 3) for v in y])
# sample balanced by super-class (2000 per class)
from numpy.random import default_rng
rng = default_rng(42)
Xs, ys = [], []
for cls in [1,2,3]:
    idx = np.flatnonzero(y_super==cls)
    take = min(2000, len(idx))
    cho = rng.choice(idx, size=take, replace=False)
    Xs.append(X[cho]); ys.append(y_super[cho])
X = np.vstack(Xs); y = np.hstack(ys)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42))
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)
print(classification_report(y_test,y_pred,target_names=["econ","water","others"]))
print(confusion_matrix(y_test,y_pred))
joblib.dump(rf,"stage1_rf_quick.joblib"); joblib.dump(scaler,"stage1_scaler_rf_quick.joblib")
