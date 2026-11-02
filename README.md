# Appendicitis-CNN — Reproducibility Toolkit (R2)

> **Research-only reference implementation — NOT a clinically validated diagnostic tool.**
>
> This repository is an R2 engineering implementation prepared to support the revision of:
> *An Ultra-Lightweight CNN Framework for Automated Acute Appendicitis Detection from CT Images with Explainable Uncertainty Estimation*, Neural Computing and Applications, manuscript `NCAA-D-25-03296R2` (revision being prepared).
>
> **Important scientific provenance**: This is not a verified restoration of the network or experiment that produced the R1 manuscript's reported 99.7% ACC, 99.2% AUC or 99.1% Dice. The previously published GitHub implementation contained `FakeData`, 256×256 inputs and a 10-epoch baseline; it could not independently support those numbers. Values in the paper must be verified using real authorized CT data, original protocols, run logs and model weights. Do not cite this code as evidence of recreating the original R1 metrics without rerunning and documenting them.

## Goals and implemented capabilities

- Custom approximately **0.58M-parameter** CNN (actual reference configuration **579,619 trainable parameters**), 224×224×3 grayscale-replicated input; SE attention, grouped 3×3 convolution (2 groups), depthwise separable 3×3/5×5 convolutions, PReLU and dropout. This is a **new, explicitly documented realization** of the architectural elements described in the R1 manuscript; stage-wise channel allocations may differ from the unpublished original implementation.
- Real image loading from authorized **2-D CT DICOM** slices or grayscale PNG/JPG/TIFF; explicit DICOM HU clipping and optional documented histogram equalization. DICOM HU conversion REQUIRES actual window bounds to be entered in YAML, rather than inventing the original protocol.
- Metadata-driven **patient-level stratified** Train/Val/Test partitions; validation rejects patient overlap or contradictory patient labels. Image-level results are aggregated to **patient-level** prior to reported classification and bootstrap estimates.
- Geometric (rotation, flip, affine, elastic deformation, crop), photometric (brightness, contrast, Gaussian noise), and batch-level CutMix/MixUp. All transforms run on **train only**. A clinical reviewer should check anatomical plausibility of generated samples before use.
- Three-stage **easy → easy+moderate → all** curriculum with mandatory verified difficulty labels. **Do not invent** case difficulty from labels or test results.
- Reproducible training (seed, YAML snapshot, early stopping based only on validation loss, best checkpoint, Python/PyTorch environment snapshot), optional class-weighted loss.
- Five torchvision baselines trained from scratch in shared protocol; optional repeat-seed runner.
- Selected Table 8-like ablation (`none`, `rotation`, `crop`, `noise`, `rotation_crop`, `rotation_crop_noise`) and expanded Table 2 leave-one-component-out ablation. Results are generated ONLY after actual training.
- MC dropout (20 passes) with validation-selected research referral threshold; predictions, uncertainty, calibration plots, ECE, Brier score, risk–coverage and optional qualitative Grad-CAM.
- Accuracy, sensitivity, specificity, precision, F1, **positive-class Dice (numerically equal to positive-class F1 for classification)**, AUROC, patient-level confusion matrices, patient-level percentile bootstrap 95% CIs and matched patient-level paired permutation tests with Holm correction.
- Actual latency benchmarking on the user's device (median/P95). CPU dynamic INT8 quantization of Linear layers is provided as a **clearly labeled experimental baseline**, not a substitute for quantization-aware training or edge-device certification.

## Quick start (Windows PowerShell; Python 3.11 recommended)

```powershell
# Run from the repository root. Use the appropriate NVIDIA CUDA PyTorch wheel.
conda create -n appendicitis-r2 python=3.11 -y
conda activate appendicitis-r2
python -m pip install -r requirements.txt
python main.py inspect-model --config configs/r2.yaml
python -m pytest -q
```

For CUDA, see the official PyTorch version selector: https://pytorch.org/get-started/locally/ .

### Real dataset manifest — required for research results

1. Create a private CSV with `image_path,patient_id,label` and optional verified `difficulty,severity,age_group,image_quality,roi_xyxy` fields. One row per authorized CT slice. For example:

```csv
image_path,patient_id,label,difficulty,severity,age_group,image_quality
D:/secure_ct/p001_slice01.dcm,study_p001,1,easy,uncomplicated,adult,good
D:/secure_ct/p002_slice01.dcm,study_p002,0,moderate,none,adult,good
```

These example identifiers and paths are placeholders, not distributed medical data.

2. Generate *patient-level stratified* split assignments, without copying data into the repository:

```powershell
python main.py split --input D:/secure_ct/manifest_raw.csv --output D:/secure_ct/manifest_split.csv --seed 2025
python main.py audit --manifest D:/secure_ct/manifest_split.csv
```

3. Update `configs/r2.yaml`: set `manifest` to your split CSV, `image_root`, **verified** `data.hu_window: [LOW, HIGH]` for DICOM (replace with your actual HU bounds), `data.histogram_equalization`, `data.target_spacing_mm`, `data.require_roi`, training and augmentation parameters matching your original script. If difficulty labels are unavailable, set `train.curriculum: false` and record that limitation. Parameters in the template are explicit engineering defaults, **not proof of what the original study used**.

4. Train and evaluate with fixed patient-level split:

```powershell
python main.py train --config configs/r2.yaml --output results/r2_main --device cuda
python main.py evaluate --checkpoint results/r2_main/checkpoint_best.pt --output results/r2_reevaluation --device cuda
```

All new experimental metrics will be computed from held-out patients and written to `results/`, which is git-ignored. The program does not contain hardcoded manuscript metrics.

### Ablation: selected Table 8 and full Table 2

```powershell
python scripts/run_ablation.py --config configs/r2.yaml --mode selected --runs 5
python scripts/run_ablation.py --config configs/r2.yaml --mode full --runs 5
```

`selected` experiments evaluate only rotation, crop and noise; **do not claim this ablates the complete pipeline**. `full` runs the complete pipeline and nine leave-one-transform-out variants; it is computationally expensive. Make sure the full pipeline is the one actually used in the submitted paper before reporting it. Outputs: per-run CSV, mean/SD summary, paired patient-level permutation tests with Holm adjustment where the reference profile is included. If multiple metrics/models are pooled into one inferential family, reapply a suitable correction across the full prespecified family.

### Baselines with comparable training protocol

```powershell
python scripts/run_baselines.py --config configs/r2.yaml --runs 5 --models r2_cnn mobilenet_v2 efficientnet_b0 efficientnet_b2 inception_v3 vgg19
```

The runner shares splits, augmentation configuration, training budget, patient-level evaluation and curriculum metadata. It **does not claim** equal model-specific hyperparameter tuning unless a separately documented, equal-budget search is completed. Training all models from scratch can be costly; a valid alternative is a matched transfer-learning protocol, documented as a different experiment.

### Audit duplicate images and leakage (on the private workstation)

```powershell
python scripts/dataset_audit.py --manifest D:/secure_ct/manifest_split.csv --image-root . --hash-files --output results/dataset_audit.json
```

The report contains aggregate counts, not patient identifiers, and flags identical image bytes appearing in different splits. The patient-level CSV audit remains mandatory; exact hashes do not detect all near-duplicates or different exports of the same underlying scan.

### Fair statistical comparison (paired patients only)

```powershell
python main.py compare --a results/r2_main/test_evaluation/patient_predictions.csv --b results/baselines/mobilenet_v2/run_1/test_evaluation/patient_predictions.csv --output results/statistical_comparison.json
```

This requires exactly matching patient IDs and labels. Default metrics: accuracy, AUC, F1. It applies Holm correction across those specified metrics for the comparison (additional hypotheses require further correction).

### Real hardware benchmark and compression probe

```powershell
python main.py benchmark --checkpoint results/r2_main/checkpoint_best.pt --device cpu --output results/benchmark_cpu
python main.py benchmark --checkpoint results/r2_main/checkpoint_best.pt --device cuda --output results/benchmark_gpu
```

On CPU it compares full precision vs *dynamic INT8 quantization of linear layers only* if supported by your PyTorch build; it does not perform QAT or structured channel removal. GPU peak allocated memory and device-specific latency are measured rather than assumed. Record CPU/GPU model, PyTorch version, quantization backend, warm-up and measurement repetitions when publishing.

### Single-image inference (research only)

```powershell
python main.py infer --checkpoint results/r2_main/checkpoint_best.pt --image D:/secure_ct/authorized_example.dcm --device cuda
```

This returns an uncalibrated model probability and MC-dropout variance, **not a clinical diagnosis**.

### Qualitative Grad-CAM

```powershell
python main.py gradcam --checkpoint results/r2_main/checkpoint_best.pt --image D:/secure_ct/authorized_example.dcm --output results/gradcam_example.png
```

The map is qualitative, not a quantitative localization metric or a radiologist reader study.

## Safe synthetic smoke test (not research!)

```powershell
python scripts/synthetic_smoke.py --output data/synthetic_smoke
# Edit a copy of config to use its manifest, epochs=1, and a small batch size.
python -m pytest -q
```

Synthetic fixture images are intentionally NOT CT. Never use the smoke-test output in a paper, rebuttal or clinical report.

## Interpretation: R1 manuscript vs R2 software

See [`R2_METHOD_ALIGNMENT.md`](R2_METHOD_ALIGNMENT.md) for a line-by-line claim-to-code map and items requiring original experimental evidence. Particularly important: original manuscript Table 3 reported 330 **test** cases whereas its Figure 9 described a matrix of 1,650 total cases; no code package can justify retaining this discrepancy. This toolkit generates results from a clearly labeled patient-level test split and preserves raw prediction records for audit.

## Repository and data policy

The `.gitignore` excludes images, DICOM scans, private manifests, checkpoints and output results. Do NOT add real medical records or labels to GitHub. Institutional approval and data access must be verified independently. This is not a certified medical device.

## Attribution

Project: **Appendicitis-CNN** (R2 reproducibility implementation). Manuscript author: Nattavut Sriwiboon, Kalasin University. New engineering implementation prepared for reproducibility and revision support. No new clinical metrics are claimed.

## License

No license has been asserted for the existing repository. Add an explicit license only after the rights holder selects one; until then, default copyright applies.
