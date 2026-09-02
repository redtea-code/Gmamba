# NACC ResNet Focal-Loss Experiment Design

## Goal

针对 NACC baseline 在类别不平衡下预测类别同质化的问题，移除无效的 `NACCMMSE`，保留疾病严重程度相关的 `CDRSUM`，使用 ResNet 和三组 Focal Loss 参数重新实验。

## Data and split

保持 seed=2026 的 subject-level split：train=174、val=45、test=57；对应标签分别为 train 141/33、val 36/9、test 47/10（negative/positive）。test 只用于最终一次评估，不用于 focal 参数选择。

NACC clinical table 的输入变量为 `CDRSUM`、`NACCGDS`、`SEX`、`AGE`；`NACCMMSE` 从 feature columns 中排除。特殊编码清洗需保持明确：`NACCMMSE` 不再输入；对仍保留的变量，不能把 NACC 特殊编码未经说明地当作正常连续值。

## Model

使用现有 baseline ResNet 结构和 MRI + generated PET 输入，保持 batch size、optimizer、learning rate、early stopping 和 seed 不变。三组实验只有 focal loss 参数不同。

Focal loss 使用 logits 稳定实现。对每个样本，`p_t` 由 logits 和 label 计算，`alpha_t` 对正类取 alpha、负类取 `1-alpha`，loss 为 `-alpha_t * (1-p_t)^gamma * log(p_t)`。

## Parameter groups

- A: `alpha=0.50`, `gamma=1.0`
- B: `alpha=0.65`, `gamma=2.0`
- C: `alpha=0.75`, `gamma=3.0`

参数选择使用 validation AUC 为主，BACC/F1/MCC 作为辅助诊断。选择出的组只在 test 上报告一次最终结果。

## Outputs

每组使用独立目录：

- `Gmamba/runs/nacc_resnet_focal_seed2026/alpha050_gamma10/`
- `Gmamba/runs/nacc_resnet_focal_seed2026/alpha065_gamma20/`
- `Gmamba/runs/nacc_resnet_focal_seed2026/alpha075_gamma30/`

每个目录记录参数、实际临床特征列表、split、训练历史、最佳 checkpoint、validation 指标、test 指标和 test predictions。

## Validation

- 单元测试验证 NACCMMSE 不进入 feature columns。
- 单元测试验证 Focal Loss 的有限值、正类加权方向和 gamma 对难例权重的影响。
- 三组 run 使用相同 seed 和 split。
- 检查每组的 prediction count，防止仅报告 accuracy 掩盖全阴性预测。
- 报告 ROC-AUC、PR-AUC（若 runner 支持）、BACC、sensitivity、specificity、F1、MCC 和 confusion matrix。
