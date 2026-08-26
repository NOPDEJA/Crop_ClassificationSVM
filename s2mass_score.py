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
import json
import os
import subprocess
import sys

import pandas as pd

M5 = "./runs/s2_2018_3date_parcel_m5"
RUN = "./runs/s2_2018_3date_parcel_s2mass"
ARMS = ("control", "treatment")
M5_TUNE = 0.2294        # the number the noise floor is measured against
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
    noise = abs(sel["control"]["tune_macro_f1"] - M5_TUNE)
    effect = sel["treatment"]["tune_macro_f1"] - sel["control"]["tune_macro_f1"]
    rows.append({"arm": "noise floor |control - M5|", "alpha2": "", "alpha3": "",
                 "cal_macro_f1": "", "tune_macro_f1": round(noise, 4),
                 "tune_alive_of_13": "", "argmax_tune_macro_f1": ""})
    rows.append({"arm": "effect treatment - control", "alpha2": "", "alpha3": "",
                 "cal_macro_f1": "", "tune_macro_f1": round(effect, 4),
                 "tune_alive_of_13": "", "argmax_tune_macro_f1": ""})
    rows.append({"arm": "claimed (|effect| > noise floor)", "alpha2": "", "alpha3": "",
                 "cal_macro_f1": "", "tune_macro_f1": str(abs(effect) > noise),
                 "tune_alive_of_13": "", "argmax_tune_macro_f1": ""})
    with open(f"{RUN}/s2mass_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    log("\n================ reading under PREDECLARATION.md ================")
    for r in rows[:3]:
        log(f"  {r['arm']:<12} cell ({r['alpha2']}, {r['alpha3']})  "
            f"tune {r['tune_macro_f1']:.4f}  ({r['tune_alive_of_13']} of 13 alive)")
    log(f"\n  noise floor |control - {M5_TUNE}| = {noise:.4f}")
    log(f"  effect      treatment - control  = {effect:+.4f}")
    log(f"  rule 3: effect is claimed only if it exceeds the noise floor -> "
        f"{'CLAIMED' if abs(effect) > noise else 'NOT CLAIMED'}")
    if noise > 0.005:
        log("  WARNING rule 5: control is more than 0.005 from M5. The pool replay is"
            " suspect and nothing should be reported until that is diagnosed.")
    log("  rule 4: fold 2 is not read, whatever the above says.")

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
