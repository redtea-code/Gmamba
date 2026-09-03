# Gmamba 实验结果总表（MCI 分类研究）

> 用途：集中记录 `Gmamba/runs` 中的实验结果，并作为后续实验逐步补填的主表。  
> 更新时间：2026-09-03  
> 结果来源：各运行目录中的 `metrics.json`；未生成该文件的运行保留在“待补充运行”中。  
> 重要说明：当前代码/历史结果的标签语义仍需核验，表中数值暂不能自动等同于 MCI 最终结果。

## 1. 指标说明

- `Val AUC`：最佳验证记录中的 AUC；`Test AUC`：锁定 checkpoint 后的测试 AUC。
- `ACC`：accuracy；`BACC`：balanced accuracy；`REC`：macro recall；`Recall`：阳性类召回率/敏感度；`PRE`：阳性类精确率；`F1`：macro F1；`MCC`：Matthews correlation coefficient。
- `—`：结果文件中不存在、数值为 NaN，或当前运行尚未完成。
- 分类结果表按运行日期升序排列；日期取自运行目录/结果路径中的 `YYYYMMDD`，无日期编码的运行置于分组末尾。
- 运行目录名是唯一实验标识；正式论文表格应进一步补充标签定义、seed、split 版本和训练日期。

## 2. 已有分类结果

### Graph-Mamba 主线

| 运行 | 数据集 | 模型/特征模式 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `adni_graphmamba_existing_pet_no_date_diff_20260817` | adni | — | full | 0.8727 | 0.7233 | 0.7258 | 0.7105 | 0.6154 | 0.6957 | 0.7132 | 0.4300 | 62.0000 | `Gmamba/runs/adni_graphmamba_existing_pet_no_date_diff_20260817/metrics.json` |
| `adni_cross_task_graphmamba_latent_concat_20260818` | adni | CrossTaskPatchGraphClassifier | full | — | — | 0.7581 | 0.7436 | 0.6538 | 0.7391 | 0.7469 | 0.4977 | 62.0000 | `Gmamba/runs/adni_cross_task_graphmamba_latent_concat_20260818/metrics.json` |
| `adni_graphconstruct_latest_gen_20260825` | adni | CrossTaskPatchGraphClassifier | full | — | — | 0.6774 | 0.6261 | 0.3077 | 0.8000 | 0.6086 | 0.3383 | 62.0000 | `Gmamba/runs/adni_graphconstruct_latest_gen_20260825/metrics.json` |
| `adni_graphconstruct_mri_only_20260825` | adni | MriPatchGraphClassifier | full | 0.8933 | 0.6880 | 0.6290 | 0.5844 | 0.3077 | 0.6154 | 0.5698 | 0.2046 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_only_20260825/metrics.json` |
| `adni_graphconstruct_mri_pet_mri_encoder_20260825` | adni | MriPetMriEncoderPatchGraphClassifier | full | 0.8440 | — | 0.6129 | 0.5545 | 0.1923 | 0.6250 | 0.5137 | 0.1604 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_mri_encoder_20260825/metrics.json` |
| `adni_graphconstruct_mri_pet_only_20260825` | adni | MriPetPatchGraphClassifier | full | 0.8662 | 0.7233 | 0.6935 | 0.6934 | 0.6923 | 0.6207 | 0.6896 | 0.3825 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_only_20260825/metrics.json` |
| `adni_graphconstruct_post2011_20260825` | adni | CrossTaskPatchGraphClassifier | full | — | — | 0.8125 | 0.6250 | 0.2500 | 1.0000 | 0.6444 | 0.4472 | 32.0000 | `Gmamba/runs/adni_graphconstruct_post2011_20260825/metrics.json` |
| `adni_graphconstruct_latest_gen_20260827_fp32` | adni | CrossTaskPatchGraphClassifier | full | 0.8818 | 0.7297 | 0.7097 | 0.6806 | 0.5000 | 0.7222 | 0.6830 | 0.3926 | 62.0000 | `Gmamba/runs/adni_graphconstruct_latest_gen_20260827_fp32/metrics.json` |
| `adni_graphconstruct_mri_pet_mri_encoder_20260827_fp32` | adni | MriPetMriEncoderPatchGraphClassifier | full | 0.8752 | 0.6966 | 0.6774 | 0.6421 | 0.4231 | 0.6875 | 0.6400 | 0.3205 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_mri_encoder_20260827_fp32/metrics.json` |
| `adni_graphconstruct_post2011_20260827_fp32` | adni | CrossTaskPatchGraphClassifier | full | 0.8426 | 0.9375 | 0.7812 | 0.5625 | 0.1250 | 1.0000 | 0.5475 | 0.3111 | 32.0000 | `Gmamba/runs/adni_graphconstruct_post2011_20260827_fp32/metrics.json` |
| `adni_graphconstruct_mri_mri_encoder_fp32_20260828_retry` | adni | MriMriEncoderPatchGraphClassifier | full | 0.8752 | 0.7511 | 0.6774 | 0.6795 | 0.6923 | 0.6000 | 0.6744 | 0.3545 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_mri_encoder_fp32_20260828_retry/metrics.json` |
| `adni_graphconstruct_mri_pet_mri_encoder_fp32_20260828` | adni | MriPetMriEncoderPatchGraphClassifier | full | 0.8752 | 0.6987 | 0.6452 | 0.5983 | 0.3077 | 0.6667 | 0.5826 | 0.2455 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_mri_encoder_fp32_20260828/metrics.json` |
| `adni_graphconstruct_mri_pet_patch2_acc_fp32_20260831` | adni | MriPetPatchGraphClassifier | full | 0.8489 | 0.6720 | 0.6613 | 0.6335 | 0.4615 | 0.6316 | 0.6338 | 0.2859 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_patch2_acc_fp32_20260831/metrics.json` |
| `adni_graphconstruct_mri_pet_patch2_extra_cnn_fp32_20260831` | adni | MriPetPatchGraphClassifier | full | 0.8785 | 0.8162 | 0.7097 | 0.7073 | 0.6923 | 0.6429 | 0.7048 | 0.4110 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_patch2_extra_cnn_fp32_20260831/metrics.json` |
| `adni_graphconstruct_mri_pet_patch2_fp32_20260831` | adni | MriPetPatchGraphClassifier | full | 0.8752 | 0.7756 | 0.6935 | 0.6934 | 0.6923 | 0.6207 | 0.6896 | 0.3825 | 62.0000 | `Gmamba/runs/adni_graphconstruct_mri_pet_patch2_fp32_20260831/metrics.json` |

### Channel-node learned-edge 新模型（旧版：未进行空间尺度对齐）

> 统一比较使用 seed=2026，与既有基线相同的 subject-level split：train=193、val=50、test=62。5 类 source：MRI、generated PET、z_rec、z_gen_mri、z_gen_pet；64 channel nodes/source；AdaptiveAvgPool3d(2,2,2)；top-k learned edge=8。

| 运行 | 数据集 | 模型/特征模式 | 模态 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `channel_graph_adni_20260901_seed2026_shared_encoder` | adni | ChannelGraphClassifier / shared_encoder | MRI + generated PET + 3 latent features + clinical | — | 0.8982 | 0.8590 | 0.7581 | 0.7489 | 0.7489 | 0.7519 | 0.7502 | 0.5008 | 62 | `Gmamba/runs/channel_graph_adni_20260901_seed2026/shared_encoder/metrics.json` |
| `channel_graph_adni_20260901_seed2026_independent_encoder` | adni | ChannelGraphClassifier / independent_encoder | MRI + generated PET + 3 latent features + clinical | — | 0.8867 | 0.8739 | 0.6452 | 0.5823 | 0.5823 | 0.7292 | 0.5367 | 0.2746 | 62 | `Gmamba/runs/channel_graph_adni_20260901_seed2026/independent_encoder/metrics.json` |

#### NACC seed=2026（旧版未对齐）

> NACC 使用同一 seed 的 subject-level split：train=174、val=45、test=57（test 类别为 47 negative / 10 positive）。

| 运行 | 数据集 | 模型/特征模式 | 模态 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `channel_graph_nacc_20260901_seed2026_shared_encoder` | nacc | ChannelGraphClassifier / shared_encoder | MRI + generated PET + 3 latent features + clinical | — | 0.8642 | 0.7149 | 0.8421 | 0.5500 | 0.5500 | 0.9196 | 0.5472 | 0.2897 | 57 | `Gmamba/runs/channel_graph_nacc_20260901_seed2026/shared_encoder/metrics.json` |
| `channel_graph_nacc_20260901_seed2026_independent_encoder` | nacc | ChannelGraphClassifier / independent_encoder | MRI + generated PET + 3 latent features + clinical | — | 0.8889 | 0.7191 | 0.8246 | 0.5000 | 0.5000 | 0.4123 | 0.4519 | 0.0000 | 57 | `Gmamba/runs/channel_graph_nacc_20260901_seed2026/independent_encoder/metrics.json` |

### Channel-node learned-edge 新模型（空间尺度已对齐）

> 已完成。四个配置均使用 seed=2026；NACC MRI/PET `(160,196,160)` 已对齐到 latent `(40,49,40)` 后进入 encoder。最后一组使用 retry2 目录。

| 运行 | 数据集 | alignment | encoder | MRI shape | latent shape | Val AUC | Test AUC | ACC | BACC | F1 | MCC | N | 结果文件 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `channel_graph_adni_20260902_alignment/adaptive_avg_pool_independent_encoder` | adni | adaptive_avg_pool | independent_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8801 | 0.8675 | 0.6452 | 0.5823 | 0.5367 | 0.2746 | 62 | `Gmamba/runs/channel_graph_adni_20260902_alignment/adaptive_avg_pool_independent_encoder/metrics.json` |
| `channel_graph_adni_20260902_alignment/adaptive_avg_pool_shared_encoder` | adni | adaptive_avg_pool | shared_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8916 | 0.8654 | 0.8065 | 0.8066 | 0.8032 | 0.6081 | 62 | `Gmamba/runs/channel_graph_adni_20260902_alignment/adaptive_avg_pool_shared_encoder/metrics.json` |
| `channel_graph_adni_20260902_alignment/learned_cnn_independent_encoder` | adni | learned_cnn | independent_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8785 | 0.8451 | 0.6935 | 0.6506 | 0.6446 | 0.3652 | 62 | `Gmamba/runs/channel_graph_adni_20260902_alignment/learned_cnn_independent_encoder/metrics.json` |
| `channel_graph_adni_20260902_alignment/learned_cnn_shared_encoder` | adni | learned_cnn | shared_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8768 | 0.7810 | 0.6129 | 0.5491 | 0.4946 | 0.1641 | 62 | `Gmamba/runs/channel_graph_adni_20260902_alignment/learned_cnn_shared_encoder/metrics.json` |
| `channel_graph_nacc_20260902_alignment/adaptive_avg_pool_independent_encoder` | nacc | adaptive_avg_pool | independent_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8858 | 0.7085 | 0.8070 | 0.4894 | 0.4466 | -0.0616 | 57 | `Gmamba/runs/channel_graph_nacc_20260902_alignment/adaptive_avg_pool_independent_encoder/metrics.json` |
| `channel_graph_nacc_20260902_alignment/adaptive_avg_pool_shared_encoder` | nacc | adaptive_avg_pool | shared_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8735 | 0.7149 | 0.8421 | 0.5500 | 0.5472 | 0.2897 | 57 | `Gmamba/runs/channel_graph_nacc_20260902_alignment/adaptive_avg_pool_shared_encoder/metrics.json` |
| `channel_graph_nacc_20260902_alignment/learned_cnn_independent_encoder_retry2` | nacc | learned_cnn | independent_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8642 | 0.7660 | 0.8246 | 0.5000 | 0.4519 | 0.0000 | 57 | `Gmamba/runs/channel_graph_nacc_20260902_alignment/learned_cnn_independent_encoder_retry2/metrics.json` |
| `channel_graph_nacc_20260902_alignment/learned_cnn_shared_encoder` | nacc | learned_cnn | shared_encoder | `(160, 196, 160)` | `(40, 49, 40)` | 0.8395 | 0.7213 | 0.8246 | 0.5000 | 0.4519 | 0.0000 | 57 | `Gmamba/runs/channel_graph_nacc_20260902_alignment/learned_cnn_shared_encoder/metrics.json` |

### 消融实验

| 运行 | 数据集 | 模型/特征模式 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

### 基线比较

| 运行 | 数据集 | 模型/特征模式 | 模态 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `adni_cnn_baseline_repro_20260831` | adni | cnn | MRI + generated PET + clinical | — | 0.8834 | 0.7639 | 0.6935 | 0.6506 | 0.6506 | 0.7214 | 0.6446 | 0.3652 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/cnn/metrics.json` |
| `adni_resnet_baseline_repro_20260831` | adni | resnet | MRI + generated PET + clinical | — | 0.9146 | 0.8889 | 0.7903 | 0.7874 | 0.7874 | 0.7847 | 0.7858 | 0.5720 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/resnet/metrics.json` |
| `adni_gcn_baseline_repro_20260831` | adni | gcn | MRI + generated PET + clinical | — | 0.8670 | 0.8226 | 0.7581 | 0.7543 | 0.7543 | 0.7519 | 0.7529 | 0.5061 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/gcn/metrics.json` |
| `adni_sgcn_baseline_repro_20260831` | adni | sgcn | MRI + generated PET + clinical | — | 0.8670 | 0.8018 | 0.6613 | 0.6549 | 0.6549 | 0.6534 | 0.6540 | 0.3084 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/sgcn/metrics.json` |
| `adni_ibgnn_baseline_repro_20260831` | adni | ibgnn | MRI + generated PET + clinical | — | 0.8801 | 0.7869 | 0.7097 | 0.7019 | 0.7019 | 0.7019 | 0.7019 | 0.4038 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/ibgnn/metrics.json` |
| `adni_mad_former_baseline_repro_20260831` | adni | mad_former | MRI + generated PET + clinical | — | 0.8424 | 0.8900 | 0.7419 | 0.7083 | 0.7083 | 0.7649 | 0.7120 | 0.4699 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/mad_former/metrics.json` |
| `adni_itcfn_baseline_repro_20260831` | adni | itcfn | MRI + generated PET + clinical | — | 0.8834 | 0.7671 | 0.5806 | 0.5000 | 0.5000 | 0.2903 | 0.3673 | 0.0000 | 62 | `Gmamba/runs/baseline_repro_adni_20260831/itcfn/metrics.json` |

#### NACC 基线复现

| 运行 | 数据集 | 模型/特征模式 | 模态 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nacc_cnn_baseline_repro_20260831` | nacc | cnn | MRI + generated PET + clinical | — | 0.8765 | 0.7617 | 0.8246 | 0.5000 | 0.5000 | 0.4123 | 0.4519 | 0.0000 | 57 | `Gmamba/runs/baseline_repro_nacc_20260831/cnn/metrics.json` |
| `nacc_resnet_baseline_repro_20260831` | nacc | resnet | MRI + generated PET + clinical | — | 0.9228 | 0.7489 | 0.8246 | 0.5000 | 0.5000 | 0.4123 | 0.4519 | 0.0000 | 57 | `Gmamba/runs/baseline_repro_nacc_20260831/resnet/metrics.json` |
| `nacc_gcn_baseline_repro_20260831` | nacc | gcn | MRI + generated PET + clinical | — | 0.8796 | 0.7319 | 0.8772 | 0.6894 | 0.6894 | 0.8423 | 0.7313 | 0.5092 | 57 | `Gmamba/runs/baseline_repro_nacc_20260831/gcn/metrics.json` |
| `nacc_sgcn_baseline_repro_20260831` | nacc | sgcn | MRI + generated PET + clinical | — | 0.8611 | 0.7213 | 0.8772 | 0.6894 | 0.6894 | 0.8423 | 0.7313 | 0.5092 | 57 | `Gmamba/runs/baseline_repro_nacc_20260831/sgcn/metrics.json` |
| `nacc_ibgnn_baseline_repro_20260831` | nacc | ibgnn | MRI + generated PET + clinical | — | 0.8951 | 0.7660 | 0.8772 | 0.6894 | 0.6894 | 0.8423 | 0.7313 | 0.5092 | 57 | `Gmamba/runs/baseline_repro_nacc_20260831/ibgnn/metrics.json` |
| `nacc_mad_former_baseline_repro_20260831` | nacc | mad_former | MRI + generated PET + clinical | — | 0.8642 | 0.7340 | 0.8246 | 0.5000 | 0.5000 | 0.4123 | 0.4519 | 0.0000 | 57 | `Gmamba/runs/baseline_repro_nacc_20260831/mad_former/metrics.json` |

### NACC ResNet + Focal Loss（移除 NACCMMSE）

> seed=2026；固定 subject-level split：train/val/test=174/45/57（test 为 47 negative / 10 positive）。输入为 MRI + generated PET + `CDRSUM`、`NACCGDS`、`SEX`、`AGE`；`NACCMMSE` 已排除。四组只改变 focal loss 参数；最佳 epoch 按 validation AUC 选择，test 仅评估一次。

| 运行 | 数据集 | 模型/损失 | alpha | gamma | Val AUC | Test AUC | ACC | BACC | F1 | MCC | Test pred+ | N | 结果文件 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nacc_resnet_focal_seed2026/alpha050_gamma10` | nacc | ResNet + focal | 0.50 | 1.0 | 0.8873 | 0.7426 | 0.8070 | 0.5287 | 0.5225 | 0.0978 | 3 | 57 | `Gmamba/runs/nacc_resnet_focal_seed2026/alpha050_gamma10/metrics.json` |
| `nacc_resnet_focal_seed2026/alpha065_gamma20` | nacc | ResNet + focal | 0.65 | 2.0 | 0.8735 | 0.6617 | 0.7544 | 0.6543 | 0.6306 | 0.2726 | 14 | 57 | `Gmamba/runs/nacc_resnet_focal_seed2026/alpha065_gamma20/metrics.json` |
| `nacc_resnet_focal_seed2026/alpha075_gamma30` | nacc | ResNet + focal | 0.75 | 3.0 | 0.8673 | 0.7745 | 0.8596 | 0.7574 | 0.7574 | 0.5149 | 10 | 57 | `Gmamba/runs/nacc_resnet_focal_seed2026/alpha075_gamma30/metrics.json` |
| `nacc_resnet_focal_seed2026/alpha085_gamma40` | nacc | ResNet + focal | 0.85 | 4.0 | 0.8704 | 0.7043 | 0.8070 | 0.6468 | 0.6526 | 0.3063 | 9 | 57 | `Gmamba/runs/nacc_resnet_focal_seed2026/alpha085_gamma40/metrics.json` |

### NACC / 外部泛化

| 运行 | 数据集 | 模型/特征模式 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nacc_binary_mri2pet_current_generator_20260820` | nacc | mri_pet | full | 0.9167 | 0.7926 | 0.8772 | 0.6894 | 0.4000 | 0.8000 | 0.7313 | 0.5092 | 57.0000 | `Gmamba/runs/nacc_binary_mri2pet_current_generator_20260820/metrics.json` |
| `nacc_graphconstruct_latent_fp32_20260828` | nacc | CrossTaskPatchGraphClassifier | full | 0.9012 | 0.7596 | 0.8772 | 0.6894 | 0.4000 | 0.8000 | 0.7313 | 0.5092 | 57.0000 | `Gmamba/runs/nacc_graphconstruct_latent_fp32_20260828/metrics.json` |
| `nacc_binary_mri2pet_native` | nacc | — | — | 0.8812 | 0.7745 | 0.8772 | 0.6894 | 0.4000 | 0.8000 | 0.7313 | 0.5092 | 57.0000 | `Gmamba/runs/nacc_binary_mri2pet_native/metrics.json` |
| `nacc_binary_mri2pet_retrained_native` | nacc | — | — | 0.8611 | 0.7298 | 0.8246 | 0.6574 | 0.4000 | 0.5000 | 0.6701 | 0.3448 | 57.0000 | `Gmamba/runs/nacc_binary_mri2pet_retrained_native/metrics.json` |
| `nacc_binary_native` | nacc | — | — | 0.8549 | 0.6894 | 0.8596 | 0.6787 | 0.4000 | 0.6667 | 0.7092 | 0.4430 | 57.0000 | `Gmamba/runs/nacc_binary_native/metrics.json` |

### 其他分类实验

| 运行 | 数据集 | 模型/特征模式 | 消融 | Val AUC | Test AUC | ACC | BACC | Recall | PRE | F1 | MCC | N | 结果文件 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `adni_resnet50_mri_pet_fp32_20260831_retry` | adni | resnet50 | — | 0.9212 | 0.8494 | 0.7742 | 0.7628 | 0.6923 | 0.7500 | 0.7654 | 0.5325 | 62.0000 | `Gmamba/runs/adni_resnet50_mri_pet_fp32_20260831_retry/metrics.json` |
| `adni_binary_mri2pet_native` | adni | — | — | 0.9072 | 0.7831 | 0.7258 | 0.7158 | 0.6538 | 0.6800 | 0.7169 | 0.4342 | 62.0000 | `Gmamba/runs/adni_binary_mri2pet_native/metrics.json` |
| `adni_binary_mri2pet_retrained_native` | adni | — | — | 0.8949 | 0.7981 | 0.7258 | 0.7212 | 0.6923 | 0.6667 | 0.7199 | 0.4402 | 62.0000 | `Gmamba/runs/adni_binary_mri2pet_retrained_native/metrics.json` |
| `adni_binary_native` | adni | — | — | 0.8998 | 0.8248 | 0.6129 | 0.5545 | 0.1923 | 0.6250 | 0.5137 | 0.1604 | 62.0000 | `Gmamba/runs/adni_binary_native/metrics.json` |

## 3. 待补充或未完成运行

| 运行目录 | 当前状态 | 下一步 |
|---|---|---|
| `adni_binary` | 有配置，缺少 metrics.json | 检查训练是否结束；若已结束，补齐 `metrics.json` 后填入上表 |
| `adni_graphconstruct_mri_mri_encoder_fp32_20260828` | 有配置，缺少 metrics.json | 检查训练是否结束；若已结束，补齐 `metrics.json` 后填入上表 |
| `adni_graphmamba_existing_pet_full_20260817` | 有配置，缺少 metrics.json | 检查训练是否结束；若已结束，补齐 `metrics.json` 后填入上表 |
| `adni_resnet50_mri_pet_fp32_20260831` | 有配置，缺少 metrics.json | 检查训练是否结束；若已结束，补齐 `metrics.json` 后填入上表 |
| `nacc_binary` | 有配置，缺少 metrics.json | 检查训练是否结束；若已结束，补齐 `metrics.json` 后填入上表 |

## 4. 生成与数据准备结果

| 运行/产物 | 类型 | 当前状态 | 关键文件 | 备注 |
|---|---|---|---|---|

## 5. 逐步填表规则

1. 每完成一次训练，确认运行目录包含 `run_config.json`、`manifest.csv`、`metrics.json` 和 `test_predictions.csv`。
2. 在对应分组增加一行：保留完整运行目录名、数据集、输入模态、消融设置和 checkpoint 选择指标。
3. 只从测试集锁定模型的 `test` 字段填入 Test 指标；验证集指标放入 Val 列，不混用。
4. 新结果注明标签定义、受试者级 split、seed 和是否使用生成 PET。

## 6. 当前主线阅读顺序

`MRI/临床基线` → `MRI + 生成 PET` → `Graph-Mamba 主模型` → `去 MRI/PET/临床消融` → `CNN/ResNet/图模型基线` → `NACC 外部泛化`。
