"""s2mass_score.py

Score the two Stage-2 arms from s2mass_stage2.py and read the result under the
rules fixed in runs/s2_2018_3date_parcel_s2mass/PREDECLARATION.md.

Nothing is scored here directly. Each arm gets a run directory whose Stage-1 and
Stage-3 artifacts are HARD LINKS to M5's -- the same bytes, not copies, so there
is no chance of a stale duplicate -- and whose only own file is that arm's
stage2_prob_val.npy. Then sweep_operating_point.py and stage2_diagnostics.py run
against those directories unmodified. That is what makes the three numbers
comparable: M5, control and treatment go through one code path, one grid, one
selection half and one scorer.

Writes runs/s2_2018_3date_parcel_s2mass/s2mass_summary.csv and
       runs/s2_2018_3date_parcel_s2mass/s2mass_routing_shift.csv
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

M5 = "./runs/s2_2018_3date_parcel_m5"
RUN = "./runs/s2_2018_3date_parcel_s2mass"
ARMS = ("control", "treatment")
# The predeclaration writes the floor as |control - 0.2294|, but 0.2294 is itself a
# rounded figure and opsweep.csv only stores four decimals, so that literal reading
# cannot resolve a floor below 1e-4. The exact scores are recomputed here at full
# precision and BOTH readings are reported, because the difference between them is
# larger than the floor itself.
M5_TUNE_ROUNDED = 0.2294
SINK = 4
GROUPS = {1: ("orchards", [2403, 2404, 2407, 2413, 2416, 2419, 2420]),
          2: ("plantation", [2302, 2303, 2405]),
          3: ("field", [2101, 2204, 2205])}
CROPS = sorted([2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405,
                2407, 2413, 2416, 2419, 2420])
SHARED = ["val_cal_idx.npy", "val_tune_idx.npy", "stage2_val_idx.npy",
          "stage1_pred.npy", "stage1_train_idx.npy", "stage1_route_oof_train.npy",
          "stage3_orchards_prob_val.npy", "stage3_plantation_prob_val.npy",
          "stage3_field_prob_val.npy", "valid_cols.npy"]


def log(*a):
    print(*a, flush=True)


def build_dir(arm):
    d = f"{RUN}/{arm}"
    os.makedirs(d, exist_ok=True)
    for f in SHARED:
        dst = f"{d}/{f}"
        if not os.path.exists(dst):
            os.link(f"{M5}/{f}", dst)
    src = f"{RUN}/stage2_{arm}_prob_val.npy"
    dst = f"{d}/stage2_prob_val.npy"
    if not os.path.exists(dst):
        os.link(src, dst)
    m5 = json.load(open(f"{M5}/manifest.json"))
    json.dump({"npz": m5["npz"], "arm": arm, "stage1_from": M5, "stage3_from": M5,
               "params_stage23": m5["params_stage23"],
               "params_stage3": m5["params_stage3"]},
              open(f"{d}/manifest.json", "w"), indent=2)
    return d


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def exact_tune_macro(run_dir, a2, a3):
    """Full-precision strict tune-half macro F1, same construction as the sweep.

    opsweep.csv rounds to four decimals, which is coarser than the quantity the
    noise floor is trying to express, so it is recomputed here rather than read.
    """
    y = np.load(json.load(open(f"{run_dir}/manifest.json"))["npz"],
                allow_pickle=True)["y"].astype(np.int32)
    cal = np.load(f"{run_dir}/val_cal_idx.npy")
    tune = np.load(f"{run_dir}/val_tune_idx.npy")
    c_va = np.load(f"{run_dir}/stage2_val_idx.npy")
    in_cal = np.isin(c_va, cal, assume_unique=True)
    c_va_cal = np.intersect1d(c_va, cal)

    g_of = np.zeros(y.size, dtype=np.int32)
    for g, (_, codes) in GROUPS.items():
        g_of[np.isin(y, codes)] = g
    g_of[g_of == 0] = SINK

    def rw(P, ratio, a):
        if a == 0.0:
            return P
        o = P * (ratio ** a)
        t = o.sum(1, keepdims=True)
        t[t == 0] = 1.0
        return o / t

    cls2 = np.array([1, 2, 3, 4])
    P2 = np.load(f"{run_dir}/stage2_prob_val.npy")
    pi2 = np.array([(g_of[c_va_cal] == g).mean() for g in cls2])
    ratio2 = ((1.0 / cls2.size) / np.where(pi2 > 0, pi2, 1e-9))[None, :]
    g_hat = cls2[rw(P2, ratio2, a2).argmax(1)]

    code = np.zeros(c_va.size, dtype=np.int32)
    for g, (gn, codes) in GROUPS.items():
        P3 = np.load(f"{run_dir}/stage3_{gn}_prob_val.npy")
        cls3 = np.array(sorted(codes))
        own = np.intersect1d(c_va_cal, np.flatnonzero(np.isin(y, codes)))
        pi = np.array([(y[own] == c).mean() for c in cls3])
        r3 = ((1.0 / len(codes)) / np.where(pi > 0, pi, 1e-9))[None, :]
        s = g_hat == g
        if s.any():
            code[s] = cls3[rw(P3[s], r3, a3).argmax(1)]

    m = ~in_cal
    pred = np.zeros(tune.size, dtype=np.int32)
    pred[np.searchsorted(tune, c_va[m])] = code[m]
    y_eval = np.where(np.isin(y[tune], CROPS), y[tune], 0)
    return float(f1_score(y_eval, pred, labels=CROPS, average="macro",
                          zero_division=0))


def run(script, d):
    log(f"\n===== {script}  RUN_DIR={d} =====")
    env = dict(os.environ, RUN_DIR=d)
    r = subprocess.run([sys.executable, "-u", script], env=env)
    if r.returncode:
        raise SystemExit(f"{script} failed on {d}")


if __name__ == "__main__":
    for arm in ARMS:
        p = f"{RUN}/stage2_{arm}_prob_val.npy"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run s2mass_stage2.py first")

    dirs = {arm: build_dir(arm) for arm in ARMS}
    for arm, d in dirs.items():
        run("sweep_operating_point.py", d)
        run("stage2_diagnostics.py", d)

    # ---------------- the three scores, one table ---------------------------
    sel = {arm: json.load(open(f"{dirs[arm]}/opsweep_selection.json")) for arm in ARMS}
    sel["M5"] = json.load(open(f"{M5}/opsweep_selection.json"))
    rows = []
    for name in ("M5", "control", "treatment"):
        s = sel[name]
        rows.append({"arm": name, "alpha2": s["alpha2"], "alpha3": s["alpha3"],
                     "cal_macro_f1": s["cal_macro_f1"],
                     "tune_macro_f1": s["tune_macro_f1"],
                     "tune_alive_of_13": s["tune_alive"],
                     "argmax_tune_macro_f1": s["baseline_tune_macro_f1"]})
    # ---------------- exact scores, and what the fit really did -------------
    ex = {n: exact_tune_macro(d, sel[n]["alpha2"], sel[n]["alpha3"])
          for n, d in [("M5", M5)] + list(dirs.items())}
    noise_exact = abs(ex["control"] - ex["M5"])
    noise_literal = abs(ex["control"] - M5_TUNE_ROUNDED)
    effect = ex["treatment"] - ex["control"]

    # The seventh check the six replay checkpoints could not make. M5 never saved
    # its Stage-2 fit indices, so the pool could not be byte-compared directly.
    # If the control's model file is byte-identical to M5's, the replay drew the
    # same rows in the same order, which is stronger evidence than any of the six.
    h5, hc = sha256(f"{M5}/stage2_model.joblib"), sha256(f"{RUN}/stage2_control_model.joblib")
    identical = h5 == hc
    links = os.stat(f"{RUN}/stage2_control_model.joblib").st_nlink

    for nm, v in (("exact tune macro F1 M5", ex["M5"]),
                  ("exact tune macro F1 control", ex["control"]),
                  ("exact tune macro F1 treatment", ex["treatment"]),
                  ("noise floor, exact |control - M5|", noise_exact),
                  ("noise floor, literal |control - 0.2294|", noise_literal),
                  ("effect treatment - control", effect)):
        rows.append({"arm": nm, "alpha2": "", "alpha3": "", "cal_macro_f1": "",
                     "tune_macro_f1": f"{v:.10f}", "tune_alive_of_13": "",
                     "argmax_tune_macro_f1": ""})
    rows.append({"arm": "claimed (effect > both floors)", "alpha2": "", "alpha3": "",
                 "cal_macro_f1": "",
                 "tune_macro_f1": str(abs(effect) > max(noise_exact, noise_literal)),
                 "tune_alive_of_13": "", "argmax_tune_macro_f1": ""})
    rows.append({"arm": "control model byte-identical to M5 (sha256)", "alpha2": "",
                 "alpha3": "", "cal_macro_f1": "", "tune_macro_f1": str(identical),
                 "tune_alive_of_13": f"hardlinks={links}",
                 "argmax_tune_macro_f1": h5[:16]})
    with open(f"{RUN}/s2mass_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    log("\n================ reading under PREDECLARATION.md ================")
    for r in rows[:3]:
        log(f"  {r['arm']:<12} cell ({r['alpha2']}, {r['alpha3']})  "
            f"tune {r['tune_macro_f1']:.4f}  ({r['tune_alive_of_13']} of 13 alive)")
    log(f"\n  exact   M5 {ex['M5']:.10f}   control {ex['control']:.10f}"
        f"   treatment {ex['treatment']:.10f}")
    log(f"  noise floor, exact   |control - M5|      = {noise_exact:.10f}")
    log(f"  noise floor, literal |control - 0.2294|  = {noise_literal:.10f}")
    log(f"  effect               treatment - control = {effect:+.10f}")
    log(f"  rule 3 -> {'CLAIMED' if abs(effect) > max(noise_exact, noise_literal) else 'NOT CLAIMED'}"
        f" (clears both readings of the floor)")
    if noise_literal > 0.005:
        log("  WARNING rule 5: control is more than 0.005 from M5. The pool replay is"
            " suspect and nothing should be reported until that is diagnosed.")
    log("  rule 4: fold 2 is not read, whatever the above says.")
    log(f"\n  seventh check: control Stage-2 model byte-identical to M5 = {identical}"
        f"  (sha256 {h5[:16]}..., hardlinks {links})")
    if identical:
        log("    So the Stage-2 fit is DETERMINISTIC given the pool, and this control")
        log("    measures the reproducibility of the SCORING path, not retrain variation.")

    # ---------------- per-crop F1 and routing accuracy ----------------------
    f1 = None
    for name, d in [("M5", M5)] + list(dirs.items()):
        t = pd.read_csv(f"{d}/opsweep_selected_tune.csv")[["crop", "support", "f1"]]
        t = t.rename(columns={"f1": name})
        f1 = t if f1 is None else f1.merge(t.drop(columns="support"), on="crop")
    log("\nper-crop tune F1 at each arm's cal-selected cell")
    log(f1.to_string(index=False))

    rt = None
    for name, d in [("M5", M5)] + list(dirs.items()):
        t = pd.read_csv(f"{d}/stage2_routing_accuracy.csv")
        t = t[t.operating_point != "argmax"][["crop", "tune_candidates",
                                              "routed_to_true_group"]]
        t = t.rename(columns={"routed_to_true_group": name})
        rt = t if rt is None else rt.merge(t.drop(columns="tune_candidates"), on="crop")
    rt["shift"] = (rt["treatment"] - rt["control"]).round(4)
    rt = rt.merge(f1.drop(columns="support"), on="crop", suffixes=("_route", "_f1"))
    rt.to_csv(f"{RUN}/s2mass_routing_shift.csv", index=False, encoding="utf-8-sig")
    log("\nStage-2 routing accuracy per crop, and the shift the treatment caused")
    log(rt.to_string(index=False))
    log(f"\nwrote {RUN}/s2mass_summary.csv and {RUN}/s2mass_routing_shift.csv")
