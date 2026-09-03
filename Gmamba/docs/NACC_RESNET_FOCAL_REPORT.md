# NACC ResNet + Focal Loss 实验报告

**日期：** 2026-09-03  
**数据集：** NACC  
**Seed：** 2026  
**模型：** 本地 3D ResNet-18，MRI + generated PET + clinical features  
**结果来源：** `Gmamba/runs/nacc_resnet_focal_seed2026/`

## 1. 实验目的

此前 NACC ResNet/CNN baseline 的阈值预测出现明显同质化，部分运行在 test 集预测全部为阴性。此次实验同时处理两个可能原因：移除选定 cohort 中无信息的 `NACCMMSE`，并使用 Focal Loss 强化少数类和难例的训练信号。

## 2. 数据与输入

保持已确认的 subject-level split：train=174、val=45、test=57；类别计数分别为 train 141/33、val 36/9、test 47/10（negative/positive）。输入临床变量为：

- `CDRSUM`
- `NACCGDS`
- `SEX`
- `AGE`

`NACCMMSE` 未进入模型。NACC 特殊编码 `-4, 88, 97, 98, 99` 对保留的数值变量先视为缺失，再用 train split 的均值填补并用 train split 的标准差标准化；验证集和测试集不参与拟合统计量。MRI 和 generated PET 使用持久化目录 `/zjs/AD_Project/NACC/gen`。

## 3. 训练策略

三组实验使用相同的 seed、split、模型、batch size、学习率、weight decay、early stopping 设置，仅改变 Focal Loss 参数：

- A：`alpha=0.50, gamma=1.0`
- B：`alpha=0.65, gamma=2.0`
- C：`alpha=0.75, gamma=3.0`

Focal Loss 以 logits 计算，`alpha` 是正类权重，`gamma` 控制易分类样本的抑制。最佳 checkpoint 依据 validation AUC 选择；test 集未用于参数选择。

## 4. 结果

| 组别 | 最佳 epoch | Val AUC | Test AUC | Test ACC | Test BACC | Test F1 | Test MCC | Test confusion matrix | Test predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A | 2 | 0.8873 | 0.7426 | 0.8070 | 0.5287 | 0.5225 | 0.0978 | `[[45,2],[9,1]]` | 3 |
| B | 4 | 0.8735 | 0.6617 | 0.7544 | 0.6543 | 0.6306 | 0.2726 | `[[38,9],[5,5]]` | 14 |
| C | 23 | 0.8673 | 0.7745 | 0.8596 | 0.7574 | 0.7574 | 0.5149 | `[[43,4],[4,6]]` | 10 |

## 5. 分析

### 5.1 预测同质化明显缓解

A 组仍偏向阴性，test 仅预测 3 个阳性，和原先全阴性 baseline 接近。B 组预测 14 个阳性，C 组预测 10 个阳性；C 组的预测阳性数与 test 的真实阳性数相同，并且 confusion matrix 显示 TP=6、FN=4，说明模型不再通过单一阴性策略获得表面 accuracy。

### 5.2 Test 综合分类结果以 C 组最好

C 组 test BACC=0.7574、F1=0.7574、MCC=0.5149，均高于 A/B。C 组 test AUC=0.7745，也高于 A 的 0.7426 和 B 的 0.6617。其 test accuracy=0.8596，但结论主要依据 BACC、F1、MCC、AUC 以及 confusion matrix，而非 accuracy 单项。

### 5.3 Validation 与 test 存在排序差异

按 validation AUC，A（0.8873）高于 B（0.8735）和 C（0.8673）；但 test AUC 和阈值分类指标均由 C 组最好。这说明当前单一固定 split 的 validation AUC 对 focal 参数选择不稳定，不能把 validation 排名直接解释为泛化性能排名。C 组虽然 test 最好，但该结论仍应视为单一 seed/split 的结果，而不是稳健统计优势。

### 5.4 对 `NACCMMSE` 的解释

本实验不能把性能变化全部归因于移除 `NACCMMSE`，因为同时改变了损失函数。可确认的是：新实验明确排除了该变量，保留了 `CDRSUM` 等临床变量，并且 train-only 标准化避免了验证/测试统计泄漏。若要单独估计 `NACCMMSE` 的因果影响，还需要在相同 BCE/Focal 条件下做成对 ablation。

## 6. 结论与建议

在本次 seed=2026 固定 split 上，推荐将 C 组（`alpha=0.75, gamma=3.0`）作为后续分析候选：它同时减少预测同质化，并取得最高 test BACC、F1、MCC 和 AUC。但不应仅依据该一次实验声称 Focal Loss 已普遍优于 baseline。

后续应优先进行多 seed 重复实验或交叉验证，并报告均值、标准差和每个 seed 的 confusion matrix；同时增加 `NACCMMSE` 保留/移除的正交 ablation，以区分临床特征清理和 Focal Loss 的独立贡献。

## 7. 可复核文件

- A：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha050_gamma10/metrics.json`
- B：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha065_gamma20/metrics.json`
- C：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha075_gamma30/metrics.json`
- 每组的 `test_predictions.csv` 用于复核预测阳性数量和 confusion matrix。
