"""diagnose_error_budget.py

Traces every economic-crop pixel through the whole cascade instead of scoring
each stage separately, and prints the four tables of S2_SVM_ANALYSIS.md 6.5:

  A  Stage-1 economic-crop recall per true crop (where the dropped pixels go)
  B  Para rubber's full fate, stage by stage
  C  For each PREDICTED crop, what was actually there  (the prior-collapse view)
  D  The end-to-end loss budget: dropped / misrouted / wrong code / correct

Scored on the held_out population, so run reconstruct_sampled_rows.py first.
Reads only saved predictions -- no model is loaded and nothing is refitted.
"""
import numpy as np, os
from config import NPZ, OUT_DIR, STAGE1_PRED, STAGE2_PRED, E2E_PRED

d = np.load(NPZ, allow_pickle=True)
y = d["y"].astype(np.int32)
s1 = np.load(STAGE1_PRED)
s2 = np.load(STAGE2_PRED)
fin = np.load(E2E_PRED)
fitted = np.load(f"{OUT_DIR}/trainval_rows_mask.npy")
held = ~fitted

LU = {2101:"Rice",2204:"Cassava",2205:"Pineapple",2302:"Rubber",2303:"OilPalm",
      2405:"Coconut",2403:"Durian",2404:"Rambutan",2407:"Mango",2413:"Longan",
      2416:"Jackfruit",2419:"Mangosteen",2420:"Langsat"}
S1N={0:"nodata",1:"econ",2:"water",3:"others",4:"forest"}
S2N={0:"-",1:"orchards",2:"plantation",3:"field",4:"other_econ"}
econ=list(LU)

print("### A. Stage-1 econ recall per crop (held_out)")
print(f"{'crop':<11}{'true_px':>10} {'->econ':>7} {'->others':>9} {'->forest':>9} {'->water':>8}")
for c in econ:
    m = held & (y==c)
    n = m.sum()
    if n==0: continue
    p = s1[m]
    print(f"{LU[c]:<11}{n:>10} {(p==1).mean():>7.3f} {(p==3).mean():>9.3f} {(p==4).mean():>9.3f} {(p==2).mean():>8.3f}")

print("\n### B. Rubber (2302) full fate breakdown, held_out")
m = held & (y==2302); n=m.sum()
print("true rubber px:", n)
lost1 = m & (s1!=1); print(f"  dropped at S1          {lost1.sum():>10} ({lost1.sum()/n:6.1%})")
for v in (3,4,2):
    k=(m&(s1==v)).sum(); print(f"      -> {S1N[v]:<9}{k:>10} ({k/n:6.1%})")
kept = m & (s1==1)
print(f"  kept as econ           {kept.sum():>10} ({kept.sum()/n:6.1%})")
for v in (1,2,3,4):
    k=(kept&(s2==v)).sum(); print(f"      S2 -> {S2N[v]:<11}{k:>7} ({k/n:6.1%})")
inplant = kept & (s2==2)
print(f"  in plantation, final code:")
for c,cnt in sorted(zip(*np.unique(fin[inplant],return_counts=True)),key=lambda t:-t[1]):
    print(f"      {LU.get(c,c):<12}{cnt:>10} ({cnt/n:6.1%})")

print("\n### C. Who eats the rubber? true-label composition of each predicted class (held_out)")
for c in econ:
    pm = held & (fin==c); tot=pm.sum()
    if tot==0: continue
    vals,cnts = np.unique(y[pm],return_counts=True)
    top = sorted(zip(vals,cnts),key=lambda t:-t[1])[:3]
    s = "  ".join(f"{LU.get(int(v),int(v))}:{k/tot:.0%}" for v,k in top)
    print(f"  pred {LU[c]:<11} n={tot:>9}  true-> {s}")

print("\n### D. loss budget over all econ (held_out)")
me = held & np.isin(y,econ); N=me.sum()
d1 = (me & (s1!=1)).sum()
sub = np.zeros_like(y,dtype=np.int8)
sub[np.isin(y,[2403,2404,2407,2413,2416,2419,2420])]=1
sub[np.isin(y,[2302,2303,2405])]=2
sub[np.isin(y,[2101,2204,2205])]=3
d2 = (me & (s1==1) & (s2!=sub)).sum()
ok = (me & (fin==y)).sum()
d3 = N-d1-d2-ok
print(f"  total econ           {N:>10}")
print(f"  dropped S1           {d1:>10} ({d1/N:6.1%})")
print(f"  misrouted S2         {d2:>10} ({d2/N:6.1%})")
print(f"  wrong code S3        {d3:>10} ({d3/N:6.1%})")
print(f"  correct              {ok:>10} ({ok/N:6.1%})")
