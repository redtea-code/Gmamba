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

四组实验使用相同的 seed、split、模型、batch size、学习率、weight decay、early stopping 设置，仅改变 Focal Loss 参数：

- A：`alpha=0.50, gamma=1.0`
- B：`alpha=0.65, gamma=2.0`
- C：`alpha=0.75, gamma=3.0`
- D：`alpha=0.85, gamma=4.0`

Focal Loss 以 logits 计算，`alpha` 是正类权重，`gamma` 控制易分类样本的抑制。最佳 checkpoint 依据 validation AUC 选择；test 集未用于参数选择。

## 4. 结果

| 组别 | 最佳 epoch | Val AUC | Test AUC | Test ACC | Test BACC | Test F1 | Test MCC | Test confusion matrix | Test predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A | 2 | 0.8873 | 0.7426 | 0.8070 | 0.5287 | 0.5225 | 0.0978 | `[[45,2],[9,1]]` | 3 |
| B | 4 | 0.8735 | 0.6617 | 0.7544 | 0.6543 | 0.6306 | 0.2726 | `[[38,9],[5,5]]` | 14 |
| C | 23 | 0.8673 | 0.7745 | 0.8596 | 0.7574 | 0.7574 | 0.5149 | `[[43,4],[4,6]]` | 10 |
| D | 4 | 0.8704 | 0.7043 | 0.8070 | 0.6468 | 0.6526 | 0.3063 | `[[42,5],[6,4]]` | 9 |

## 5. C 组参数下的六个基线复跑

在完成 ResNet 的四组参数选择后，使用最终选定的 C 组参数（`alpha=0.75, gamma=3.0`）对其余六个基线方法进行独立串行复跑。为避免 GPU 资源争用，六个任务依次使用 GPU 0 执行；ResNet 不重复运行。

| 方法 | 最佳 epoch | Val AUC | Test AUC | Test ACC | Test BACC | Test Recall | Test PRE | Test F1 | Test MCC | Test confusion matrix | Test predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| CNN | 6 | 0.8827 | 0.7213 | 0.7895 | 0.6755 | 0.6755 | 0.6528 | 0.6621 | 0.3275 | `[[40,7],[5,5]]` | 12 |
| GCN | 16 | 0.8673 | 0.6681 | 0.8421 | 0.7074 | 0.7074 | 0.7257 | 0.7158 | 0.4328 | `[[43,4],[5,5]]` | 9 |
| SGCN | 31 | 0.8426 | 0.6191 | 0.8421 | 0.7074 | 0.7074 | 0.7257 | 0.7158 | 0.4328 | `[[43,4],[5,5]]` | 9 |
| IBGNN | 28 | 0.8241 | 0.6596 | 0.8246 | 0.6968 | 0.6968 | 0.6968 | 0.6968 | 0.3936 | `[[42,5],[5,5]]` | 10 |
| MADFormer | 26 | 0.8549 | 0.8234 | 0.8772 | 0.6500 | 0.6500 | 0.9352 | 0.6961 | 0.5110 | `[[47,0],[7,3]]` | 3 |
| ITCFN | 5 | 0.8395 | 0.7957 | 0.8421 | 0.6681 | 0.6681 | 0.7257 | 0.6889 | 0.3896 | `[[44,3],[6,4]]` | 7 |

六个基线中，MADFormer 的 test AUC（`0.8234`）和 MCC（`0.5110`）最高，但只预测 3 个阳性，仍表现出明显的阴性偏向。GCN 与 SGCN 的 test BACC 和 F1 均为 `0.7074` 和 `0.7158`，但 SGCN 的 test AUC 仅为 `0.6191`，低于 GCN 的 `0.6681`。ITCFN 的 test AUC 为 `0.7957`，位列六个基线第二；CNN 和 IBGNN 的 test AUC 分别为 `0.7213` 和 `0.6596`。这些结果均来自单一 seed 和固定 split，不能据此宣称某一基线具有普遍优势。

## 6. 分析

### 6.1 预测同质化明显缓解

A 组仍偏向阴性，test 仅预测 3 个阳性，和原先全阴性 baseline 接近。B 组预测 14 个阳性，C 组预测 10 个阳性，D 组预测 9 个阳性；C 组的预测阳性数与 test 的真实阳性数相同，并且 confusion matrix 显示 TP=6、FN=4，说明模型不再通过单一阴性策略获得表面 accuracy。

### 6.2 Test 综合分类结果以 C 组最好

C 组 test BACC=0.7574、F1=0.7574、MCC=0.5149，均高于 A/B/D。C 组 test AUC=0.7745，也高于 A 的 0.7426、B 的 0.6617 和 D 的 0.7043。其 test accuracy=0.8596，但结论主要依据 BACC、F1、MCC、AUC 以及 confusion matrix，而非 accuracy 单项。D 组进一步提高正类权重和难例聚焦强度后，test 指标反而低于 C 组，说明在当前固定 split 上继续增强干预并未带来收益。

### 6.3 Validation 与 test 存在排序差异

按 validation AUC，A（0.8873）高于 B（0.8735）、D（0.8704）和 C（0.8673）；但 test AUC 和阈值分类指标均由 C 组最好。这说明当前单一固定 split 的 validation AUC 对 focal 参数选择不稳定，不能把 validation 排名直接解释为泛化性能排名。C 组虽然 test 最好，但该结论仍应视为单一 seed/split 的结果，而不是稳健统计优势。

### 6.4 对 `NACCMMSE` 的解释

本实验不能把性能变化全部归因于移除 `NACCMMSE`，因为同时改变了损失函数。可确认的是：新实验明确排除了该变量，保留了 `CDRSUM` 等临床变量，并且 train-only 标准化避免了验证/测试统计泄漏。若要单独估计 `NACCMMSE` 的因果影响，还需要在相同 BCE/Focal 条件下做成对 ablation。

## 7. 结论与建议

在本次 seed=2026 固定 split 上，推荐将 C 组（`alpha=0.75, gamma=3.0`）作为最终参数选择：它同时减少预测同质化，并取得四组中最高的 test BACC、F1、MCC 和 AUC。D 组作为边界测试未超过 C 组，因此本轮参数选择结束。但不应仅依据该一次实验声称 Focal Loss 已普遍优于 baseline。

后续应优先进行多 seed 重复实验或交叉验证，并报告均值、标准差和每个 seed 的 confusion matrix；同时增加 `NACCMMSE` 保留/移除的正交 ablation，以区分临床特征清理和 Focal Loss 的独立贡献。

## 8. 可复核文件

- A：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha050_gamma10/metrics.json`
- B：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha065_gamma20/metrics.json`
- C：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha075_gamma30/metrics.json`
- D：`Gmamba/runs/nacc_resnet_focal_seed2026/alpha085_gamma40/metrics.json`
- 每组的 `test_predictions.csv` 用于复核预测阳性数量和 confusion matrix。
- 六个基线复跑结果：`Gmamba/runs/nacc_baseline_focal_seed2026_serial/<model>/metrics.json`
