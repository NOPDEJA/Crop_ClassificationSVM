"""compare_stage1_arms.py

Stage 1 only, several arms, one population.

    python compare_stage1_arms.py s2_2018_3date s2_2018_3date_v2 s2_2018_5date

Arms do NOT share a train/val split: each drops different rows for NaN and samples its own
stages. 514,174 pixels held out by the 3-date arm are rows the 5-date arm trained on, and
they are enriched in Stage-2/3 samples -- which are drawn conditional on Stage 1 having
routed them to econ, so an arm scores econ recall 1.000 on its own by construction. Scoring
one arm on another arm's held_out therefore inflates it, and most for the rare crops, which
lose the largest share of themselves to sampling. The population below is the intersection:
held out by every arm compared.
"""
import numpy as np, pandas as pd, sys
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
from evaluate_flat_15class import water_code, forest_code, CROPS

ARMS = {"s2_2018_3date":"s2_3date", "s2_2018_5date":"s2_5date", "s2_2018_3date_v2":"s2_3date_v2"}
NPZ = {"s2_2018_3date":"./aligned_features/svm_s2_3date_features_labels.npz",
       "s2_2018_5date":"./aligned_features/svm_s2_only_features_labels.npz",
       "s2_2018_3date_v2":"./aligned_features/svm_s2_3date_features_labels.npz"}
NAME={1:"econ",2:"water",3:"others",4:"forest"}

def super_of(y):
    s=np.full(y.size,3,dtype=np.int8)
    s[np.isin(y,list(CROPS))]=1
    s[np.isin(y,list(water_code))]=2
    s[np.isin(y,list(forest_code))]=4
    return s

# Population = held out by EVERY arm compared, built from the arms actually named. Arms
# never share a split, even on the same NPZ with the same seed: stages 2 and 3 train on the
# pixels STAGE 1 ROUTED to econ, so the moment Stage 1 differs the sampled rows differ too.
# s2_2018_3date and s2_2018_3date_v2 differ only in the PCA removal and still diverge by
# ~506,000 rows in each direction. Those rows are selected conditional on correct routing,
# so an arm scores econ recall 1.000 on its own by construction -- scoring one arm on
# another's held_out inflates it, worst for the rare crops, which lose the largest share of
# themselves to sampling.
MASK = ~np.logical_or.reduce([np.load(f"./runs/{a}/trainval_rows_mask.npy") for a in sys.argv[1:]])
print("common held-out population:", int(MASK.sum()))
rows=[]; per=[]
for arm in sys.argv[1:]:
    tag=ARMS[arm]
    y=np.load(NPZ[arm],allow_pickle=True)["y"].astype(np.int32)
    p=np.load(f"./runs/{arm}/stage1_{tag}_pred.npy")
    m=MASK
    t=super_of(y)[m]; q=p[m]
    r=classification_report(t,q,output_dict=True,zero_division=0)
    rows.append({"arm":arm,"n":int(m.sum()),"accuracy":round(accuracy_score(t,q),4),
                 "kappa":round(cohen_kappa_score(t,q),4),
                 "f1_macro":round(r["macro avg"]["f1-score"],4)})
    for k,n in NAME.items():
        d=r[str(k)]; per.append({"arm":arm,"class":n,"precision":round(d["precision"],4),
                                 "recall":round(d["recall"],4),"f1":round(d["f1-score"],4),
                                 "support":int(d["support"])})
pd.set_option("display.width",200)
print(pd.DataFrame(rows).to_string(index=False)); print()
print(pd.DataFrame(per).pivot(index="class",columns="arm",values="f1").to_string())
