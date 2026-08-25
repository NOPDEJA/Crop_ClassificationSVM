# GPU SVM options for the S2-only 3-date / 5-date experiment

**Checked:** 2026-08-26  
**Target machine:** Windows, NVIDIA GeForce RTX 4070 (12,282 MiB), driver 591.86 / CUDA 13.1, native Python 3.13.5  
**Scope:** the current root-level pipeline only; nothing in `2018/` should be changed.

## Recommendation

Use **RAPIDS cuML 26.08 in Ubuntu under WSL2**, but separate two questions:

1. **For the lowest-risk trial**, retain the parcel-disjoint `Nystroem(RBF) -> LinearSVC` estimator and run a `cuml.accel` smoke/parity check on the current 3-date arm first. Only proceed to the 5-date run if the profile confirms useful GPU work and the 3-date result remains within a predeclared metric tolerance.
2. **For a native GPU port**, make the simplest structural speed fix first: transform once with a shared imputer/scaler/Nyström map, then use one multiclass-OvR `LinearSVC` instead of wrapping the entire pipeline in `OneVsRestClassifier`. Implement the Nyström map plus `cuml.svm.LinearSVC` in a separate arm, then train **both complete 3-date and 5-date arms on that same backend**. A GPU 5-date result is not a controlled comparison with the existing CPU 3-date result.
3. **For an exact RBF SVM experiment**, use native `cuml.svm.SVC` on bounded, identical stratified samples from the two arms. Treat this as a new model family, not a faster reproduction of the current model. Do not attempt the full Stage-1 population first.

Do **not** invest in ThunderSVM for this machine unless cuML is blocked. Its packaged ecosystem is too stale for Python 3.13 and current CUDA.

## Why this is the right next experiment

The active model is not a full kernel `SVC`. All three stages use scikit-learn's approximate RBF map followed by a linear SVM, wrapped one-vs-rest and calibrated. The corrected current implementation is in [`train_parcel_cascade.py`](../train_parcel_cascade.py); it fits the base once on parcel fold 0 and fits per-class sigmoids once on disjoint fold 1.

The project already has completed older pixel-split 3-date and 5-date arms. The documented base parcel-disjoint 3-date result is macro F1 0.2248, and the later `m0` arm reaches 0.2283. The latest clean 3-date arm is `s2_2018_3date_parcel_m5`: 30 S2-only features, `alpha2=0.2`, `alpha3=0.7`, hard-routing macro F1 **0.2344** and weighted F1 **0.7974** on 5,500,269 test pixels. A clean parcel-disjoint five-date counterpart is still missing. The shared parcel split, feature recipe, sampling policy, model capacity, calibration, and inference rule must be matched when testing whether March/April add information.

| Item | 3 dates | 5 dates |
|---|---:|---:|
| Original date-stack features | 24 | 40 |
| NPZ size on disk | 2.38 GB | 3.94 GB |
| Correct comparison | same parcel rows/split | same parcel rows/split |

The current best 3-date arm has 30 columns because it adds S2-only temporal features to the original 24-column date stack. A publication-quality 3-versus-5 comparison must either use the original 24/40 feature recipe in both arms or construct the analogous temporal features for both; it must not silently compare different feature recipes.

The corrected Stage-1 base fit uses about 2.88M rows and 250 Nyström components. Its transformed block is about 2.9 GB as `float32`, or 5.8 GB as `float64`. Stages 2/3 use 600 components but far fewer rows. Thus the 12 GB card can plausibly fit **one float32 transformed fit at a time**, but it has too little margin for parallel CV clones or several full transforms.

Five dates do not by themselves make GPU memory infeasible: at 1M rows, raw `float32` input is about 96 MB for 24 columns and 160 MB for 40. Row count, Nyström components, calibration/CV replication, and support-vector count matter much more.

## What cuML can and cannot do here

### Low-risk smoke test: `cuml.accel`

`cuml.accel` can intercept compatible scikit-learn estimators and works with pipelines and `RandomizedSearchCV`. Its current compatibility table supports dense `LinearSVC` with OvR, while unsupported estimators fall back to CPU. `Nystroem` is absent from the accelerated-estimator inventory, so the reasonable expectation is that the RBF feature map remains on CPU; verify rather than assume with the official profiler. [cuML accelerator overview](https://docs.rapids.ai/api/cuml/stable/cuml-accel/), [compatibility table](https://docs.rapids.ai/api/cuml/stable/cuml-accel/compatibility/), [profiling guide](https://docs.rapids.ai/api/cuml/stable/cuml-accel/logging-and-profiling/)

Run a separate-output smoke test in WSL:

```bash
ARM=s2_2018_3date \
ARM_OUT=./runs/gpu_smoke_3date \
SMOKE=1 \
python -m cuml.accel --profile train_parcel_cascade.py
```

Use `-v` as well when a CPU fallback needs an explanation. Keep `n_jobs=1`; the profiler does not see subprocess GPU calls, and concurrent candidates would compete for 12 GB. A profiler run measures dispatch, not trustworthy wall time, so benchmark a second unprofiled run after GPU/JIT warm-up.

This may deliver only a modest gain because the repeated CPU Nyström transforms are likely the bottleneck. That result would be useful, not a failure: it tells us a zero-change GPU port is not the solution.

### Deeper GPU port: GPU Nyström plus `cuml.svm.LinearSVC`

The best engineering prototype is:

1. Fit the imputer, scaler, and an explicitly validated GPU Nyström map on exactly the current fold and fixed hyperparameters.
2. Materialize the transformed data once as `float32` and send it to native `cuml.svm.LinearSVC(multi_class="ovr")`.
3. Keep the existing one-pass Platt sigmoids on the untouched validation fold and keep chunked prediction.

The current `OneVsRestClassifier(base_pipe(...))` clones and refits the seeded imputer/scaler/Nyström pipeline for every binary class. `LinearSVC` already implements multiclass OvR, so the wrapper need only surround the classifier—not the expensive feature transform—or can be removed in favor of one multiclass `LinearSVC`. Computing one shared feature map removes redundant transforms even before GPU acceleration. This is a local-code inference, and moving the estimator seam is not automatically equivalent: require smoke parity on fixed rows, then compare full held-out metrics. Native cuML `LinearSVC` supports multiclass OvR and balanced class weights, although the corrected parcel run should retain `class_weight=None` to preserve its protocol. [cuML LinearSVC API](https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.svm.linearsvc/)

Because the cuML solver is not numerically identical to scikit-learn's, compare model quality, not coefficients or exact predictions. NVIDIA gives the same guidance for accelerated estimators. [cuML result-compatibility guidance](https://docs.rapids.ai/api/cuml/stable/cuml-accel/compatibility/)

### Exact GPU RBF: direct `cuml.svm.SVC`

Native cuML `SVC` supports RBF kernels, `class_weight`, and multiclass OvO/OvR. Its default 1024 MiB `cache_size` bounds the kernel cache during fitting and the temporary prediction buffer; the latter can still be large when there are many support vectors. [cuML SVC API](https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.svm.svc/)

Important limits:

- Do not use a precomputed Gram matrix. A dense `float32` matrix is `4*n^2` bytes: about 10 GB at 50k rows and 40 GB at 100k.
- A bounded cache prevents that full allocation, but it does not make exact RBF training cheap. Scikit-learn documents at-least-quadratic fit scaling and says kernel SVC may be impractical beyond tens of thousands of rows. That scaling warning applies to the algorithmic experiment even though cuML's GPU solver is faster. [scikit-learn SVC scaling guidance](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- `cuml.accel` is not the route for this test: its sklearn `SVC` proxy falls back to CPU for multiclass targets. Instantiate `cuml.svm.SVC` directly.
- In cuML 26.08, direct `SVC` no longer exposes native probability fitting/`predict_proba`. Use decision scores plus the project's existing disjoint-validation sigmoids. cuML recommends external `CalibratedClassifierCV(..., ensemble=False)` generally, and scikit-learn also supports calibrating an already-fitted model via `FrozenEstimator`; the local manual sigmoid path is cheaper and already handles the rare-class edge cases. [cuML SVC source](https://github.com/NVIDIA/cuml/blob/133c6e294/python/cuml/cuml/svm/svc.py), [scikit-learn calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)

Suggested exact-SVC ladder, using the same selected rows for both date arms:

1. 25k rows, fixed `C`/`gamma`, `float32`, `cache_size=1024` or 2048 MiB.
2. 50k, then 100k only if fit time, peak VRAM, and number of support vectors remain acceptable.
3. Test Stage 3 first if the goal is a useful full model: the corrected capped base-fit counts are 172,371 orchard rows, 148,344 plantation rows, and 210,000 field rows, versus about 2.88M at Stage 1.
4. Compare parcel-disjoint macro F1, weighted F1, per-crop F1, calibration/log loss, fit time, prediction time, and peak VRAM. Do not promote it on speed alone.

If the ladder succeeds, run the complete hierarchy twice on the GPU backend—once with the 3-date matrix and once with the matched 5-date matrix. Sample-only results establish feasibility; they do not answer the scientific date-window question.

## Windows / Python installation reality

RAPIDS 26.08 supports Python 3.11-3.14, CUDA 13.0-13.3 with driver 580+, and Ada compute capability 8.9. The RTX 4070/driver combination qualifies. Windows is supported **through WSL2**, not as a native Windows Python installation. RAPIDS recommends 16 GB or more GPU RAM for WSL, so 12 GB is supported hardware but below its comfort recommendation. [RAPIDS platform matrix](https://docs.rapids.ai/platform-support/), [RAPIDS WSL installation guide](https://docs.rapids.ai/install/#windows-wsl2)

Local setup status: only the stopped `docker-desktop` WSL distribution is present, the Docker daemon is stopped, and no Ubuntu WSL distribution is installed. GPU training therefore needs environment setup before either cuML path can run.

Practical setup rules:

- Install/update Ubuntu WSL2, then create a separate RAPIDS environment using the official release selector. Do not modify the native project's Python environment.
- Keep the Windows NVIDIA driver. Do **not** install a Linux display driver inside WSL; install only the WSL/Linux CUDA toolkit or use the RAPIDS container/Conda route selected by NVIDIA. WDDM is the expected mode for GeForce CUDA access through WSL. [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- Match the RAPIDS package suffix/environment to CUDA 13 (or deliberately choose a supported CUDA 12 environment); `nvidia-smi`'s CUDA line is a driver capability, not proof that a matching toolkit is installed.
- Leave GPU candidates serial and use `float32`. `cuml.accel` does not enable managed memory on WSL2, and NVIDIA documents limited full managed-memory/oversubscription support there, so do not plan on spilling transparently beyond 12 GB. [cuML memory behavior](https://docs.rapids.ai/api/cuml/stable/cuml-accel/usage/), [CUDA WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/#known-limitations-for-linux-cuda-applications)
- Record the full environment beside every model. cuML warns that serialized estimator compatibility is not guaranteed across dependency versions. [cuML serialization FAQ](https://docs.rapids.ai/api/cuml/stable/cuml-accel/faq/)

## ThunderSVM assessment

ThunderSVM remains a credible historical GPU SVM design: its original paper reported order-of-magnitude speedups over LIBSVM and the library supports SVC/probabilistic SVM interfaces. [Original JMLR paper](https://jmlr.org/papers/v19/17-740.html)

It is a poor fit for this workstation now. PyPI's latest release is 0.3.12 from March 2020 (uploaded from CPython 3.6); the official binaries/documentation target CUDA 9 on Linux and CUDA 10 on Windows. The source tree received a few fixes through April 2024, but Python 3.13 + current CUDA compatibility is not provided as a tested package. A manual source build would add risk without evidence that it beats current cuML on Ada. [PyPI release history](https://pypi.org/project/thundersvm/), [official repository/install matrix](https://github.com/Xtra-Computing/thundersvm), [official commit history](https://github.com/Xtra-Computing/thundersvm/commits/master/)

Scholarly searches also surfaced the authors' later probabilistic multiclass GPU SVM work. Its main engineering result—batching and kernel/support-vector reuse are necessary because naive parallel probability estimation exceeds GPU memory—supports the conservative serial-calibration plan above, but does not make the old ThunderSVM packaging a better deployment choice. [Original IEEE paper](https://doi.org/10.1109/TKDE.2018.2866097)

## Go / no-go criteria

Proceed from `cuml.accel` to the hybrid port only if the profile confirms GPU `LinearSVC` execution and either shows useful speedup or identifies CPU Nyström as the remaining dominant cost. Proceed from 50k to larger exact-SVC samples only while peak use stays comfortably below 12 GB and support-vector growth/prediction time remain controlled.

For either port, acceptance requires the same parcel split, frozen sample indices and hyperparameters, no test-fold calibration, all expected classes present, and no material metric regression against the corresponding CPU arm. The sequence is:

1. Validate `cuml.accel` on the 3-date smoke/full parity case, then run the pending matched 5-date arm only if parity is acceptable.
2. If building a native GPU Nyström/linear-SVM or exact-SVC backend, run **both full 3-date and full 5-date arms** with that backend before drawing any date-window conclusion.

Do not compare a native-GPU five-date model directly with the retained CPU three-date model; backend and date count would be confounded.
