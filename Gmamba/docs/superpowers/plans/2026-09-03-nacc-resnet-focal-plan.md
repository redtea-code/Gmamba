# NACC ResNet Focal-Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正 NACC 临床输入并使用 ResNet 与三组 Focal Loss 参数完成可复现的类别不平衡实验。

**Architecture:** 在 baseline runner 中增加 NACC-specific feature exclusion，使 `NACCMMSE` 不进入模型、`CDRSUM` 保留；将 numeric 标准化改为 train-only 统计。新增 logits-stable `BinaryFocalLoss`，训练和验证使用相同 criterion，三组配置保持相同 seed/split/model，仅改变 alpha/gamma。

**Tech Stack:** Python, PyTorch, pandas, scikit-learn, pytest.

## Global Constraints

- NACC seed=2026 split remains train=174, val=45, test=57 with labels 141/33, 36/9, 47/10.
- NACCMMSE is excluded; CDRSUM remains.
- Focal groups are alpha/gamma `(0.50,1.0)`, `(0.65,2.0)`, `(0.75,3.0)`.
- Test is evaluated only after validation-based group selection.
- Preserve original baseline outputs and existing non-NACC behavior.
- Record feature list, focal parameters, seed, split, and prediction counts in metadata.

---

### Task 1: Add failing focal loss and clinical feature tests

**Files:**
- Modify: `Gmamba/test_nacc_data_audit.py` (create if absent)
- Test: `Gmamba/test_nacc_data_audit.py`

**Interfaces:**
- `BinaryFocalLoss(alpha: float, gamma: float, reduction: str = "mean") -> Tensor`.
- `prepare_clinical_features(table, dataset, fit_indices=None) -> dict` exposes `numeric_columns` and train-only normalization metadata.

- [ ] **Step 1: Write tests**

Test that `BinaryFocalLoss(alpha=..., gamma=...)` returns finite scalar loss, increasing positive alpha increases the loss contribution of a positive example, and `gamma=0` matches weighted BCE. Test that NACC feature columns exclude `NACCMMSE` and include `CDRSUM`, `NACCGDS`, `SEX`, `AGE`.

- [ ] **Step 2: Run RED**

```bash
conda run -n cjw python -m pytest Gmamba/test_nacc_data_audit.py -q
```

Expected: failure because the loss class and NACC exclusion behavior are absent.

- [ ] **Step 3: Commit test-only changes**

```bash
git add Gmamba/test_nacc_data_audit.py
git commit --no-verify -m "test: specify NACC focal loss and clinical features"
```

### Task 2: Implement focal loss and NACC clinical correction

**Files:**
- Modify: `Gmamba/run_baseline_comparison.py`
- Modify: `Gmamba/run_ad_project_binary_repro.py` if it is the selected ResNet runner
- Test: `Gmamba/test_nacc_data_audit.py`

**Interfaces:**
- `BinaryFocalLoss.forward(logits: Tensor, targets: Tensor) -> Tensor` accepts logits and binary float targets.
- NACC clinical preparation excludes only `NACCMMSE`; `CDRSUM` remains.

- [ ] **Step 1: Implement logits-stable loss**

Use `bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")`, `p_t = exp(-bce)`, `alpha_t = alpha*targets + (1-alpha)*(1-targets)`, and return `alpha_t * (1-p_t).pow(gamma) * bce`, reduced according to `reduction`.

- [ ] **Step 2: Exclude NACCMMSE**

After identifying table feature columns, for dataset `nacc` remove the exact `NACCMMSE` column while retaining `CDRSUM`, `NACCGDS`, `SEX`, and `AGE`. Record `excluded_columns=["NACCMMSE"]`.

- [ ] **Step 3: Make numeric normalization train-only**

Fit mean/std using matched train rows only, store them in metadata, and apply them to val/test. Do not alter label or subject split logic.

- [ ] **Step 4: Run GREEN**

```bash
conda run -n cjw python -m pytest Gmamba/test_nacc_data_audit.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit implementation**

```bash
git add Gmamba/run_baseline_comparison.py Gmamba/run_ad_project_binary_repro.py Gmamba/test_nacc_data_audit.py
git commit --no-verify -m "feat: add NACC focal loss and clinical cleanup"
```

### Task 3: Add ResNet focal experiment runner

**Files:**
- Create: `Gmamba/run_nacc_resnet_focal.py`
- Test: `Gmamba/test_nacc_data_audit.py`

**Interfaces:**
- CLI options `--alpha`, `--gamma`, `--seed`, `--output-dir`.
- Runner uses local `ResNetEncoder`, MRI + generated PET, clinical features without NACCMMSE.
- Metadata includes `model="resnet"`, `focal_loss`, `clinical_columns`, `splits`, `seed`.

- [ ] **Step 1: Add runner smoke test**

Construct the runner model on synthetic tensors and assert finite logits with shape `[B,1]`; assert parser accepts all three alpha/gamma groups.

- [ ] **Step 2: Run RED**

```bash
conda run -n cjw python -m pytest Gmamba/test_nacc_data_audit.py -q
```

Expected: runner import or parser test fails before the new file exists.

- [ ] **Step 3: Implement the thin runner**

Reuse existing NACC discovery, seed split, native MRI/PET loading, ResNet model construction, evaluation and artifact format. Add `BinaryFocalLoss` to train/validation. Keep test evaluation after training and write test predictions with probability and prediction columns.

- [ ] **Step 4: Run focused tests and help**

```bash
conda run -n cjw python -m pytest Gmamba/test_nacc_data_audit.py -q
conda run -n cjw python Gmamba/run_nacc_resnet_focal.py --help
```

Expected: tests pass and help shows alpha/gamma options.

- [ ] **Step 5: Commit runner**

```bash
git add Gmamba/run_nacc_resnet_focal.py Gmamba/test_nacc_data_audit.py
git commit --no-verify -m "feat: add NACC ResNet focal runner"
```

### Task 4: Run and select three focal configurations

**Files:**
- Outputs: `Gmamba/runs/nacc_resnet_focal_seed2026/alpha050_gamma10/`
- Outputs: `Gmamba/runs/nacc_resnet_focal_seed2026/alpha065_gamma20/`
- Outputs: `Gmamba/runs/nacc_resnet_focal_seed2026/alpha075_gamma30/`

- [ ] **Step 1: Run configuration A**

```bash
conda run -n cjw python Gmamba/run_nacc_resnet_focal.py --alpha 0.50 --gamma 1.0 --seed 2026 --output-dir Gmamba/runs/nacc_resnet_focal_seed2026/alpha050_gamma10
```

- [ ] **Step 2: Run configuration B**

Use the same command with `--alpha 0.65 --gamma 2.0` and the B output directory.

- [ ] **Step 3: Run configuration C**

Use the same command with `--alpha 0.75 --gamma 3.0` and the C output directory.

- [ ] **Step 4: Verify artifacts**

Each directory must contain `run_config.json`, `metrics.json`, `test_predictions.csv`, `history.jsonl`, `best.pt`, and `last.pt`; metadata must list `clinical_columns` without `NACCMMSE` and `seed=2026`.

- [ ] **Step 5: Select by validation**

Compare only validation AUC first, then validation BACC/F1/MCC as diagnostics. Select one group without using test metrics; report all three test metrics separately for transparent comparison and identify the selected group.

### Task 5: Update documentation and verify

**Files:**
- Modify: `Gmamba/EXPERIMENT_RESULTS_MASTER.md`
- Create: `Gmamba/docs/NACC_RESNET_FOCAL_REPORT.md`

- [ ] **Step 1: Add three results to a dedicated focal-loss section**

Record alpha, gamma, clinical columns, validation/test metrics, confusion matrix and positive prediction count. Keep prior baseline rows unchanged and label them as pre-cleanup BCE baseline.

- [ ] **Step 2: Write interpretation**

Explain whether focal loss reduces all-negative prediction, whether `CDRSUM` remains useful, and whether validation-selected parameters transfer to test. Do not claim improvement from accuracy alone.

- [ ] **Step 3: Run final verification**

```bash
conda run -n cjw python -m pytest Gmamba/test_nacc_data_audit.py -q
```

Check every result file against its metadata and verify `NACCMMSE` is absent from all three new configs.
