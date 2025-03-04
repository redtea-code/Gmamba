# AFF-GMamba
This code is the pytorch implementation of the AFF-GMamba.
Through the tabel file, our dataset can be reproduced and We will provide a link to our weight file in the near future.
![pic](./assets/architecture.png)

## Metrics of AD prediction on SOTA models 
| Dataset |        Method        |    ACC    |    REC    |    PRE    |    AUC    |    F1     |
| ------- | :------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| ADNI    |         GCN          |   0.672   |   0.667   |   0.669   |   0.681   |   0.667   |
|         |        ResNet        |   0.721   |   0.720   |   0.719   |   0.729   |   0.720   |
|         |        IBGNN         |   0.737   |   0.735   |   0.735   |   0.781   |   0.735   |
|         |         SGCN         |   0.786   |   0.792   |   0.792   |   0.791   |   0.786   |
|         |         CNN          |   0.639   |   0.634   |   0.636   |   0.658   |   0.634   |
|         |      MAD-former      |   0.819   |   0.819   |   0.818   |   0.847   |   0.818   |
|         |        ITCFN         |   0.868   |   0.821   |   0.884   |   0.916   |   0.851   |
|         | **AFF-Mamba (Ours)** | **0.901** | **0.821** | **0.958** | **0.951** | **0.884** |
| OASIS-3 |         GCN          |   0.620   |   0.630   |   0.699   |   0.695   |   0.589   |
|         |        ResNet        |   0.655   |   0.666   |   0.791   |   0.814   |   0.618   |
|         |        IBGNN         |   0.689   |   0.697   |   0.747   |   0.807   |   0.675   |
|         |         SGCN         |   0.724   |   0.733   |   0.818   |   0.810   |   0.707   |
|         |         CNN          |   0.655   |   0.664   |   0.724   |   0.755   |   0.633   |
|         |      MAD-former      |   0.758   |   0.761   |   0.769   |   0.845   |   0.730   |
|         |        ITCFN         |   0.689   |   0.866   |   0.650   |   0.764   |   0.742   |
|         |   **AFF-Mamba (Ours)**   |   **0.862**   |   **0.866**   |   **0.866**   |   **0.900**   |   **0.866**   |

## Pre-requisties
- Linux

- python==3.10
    
- NVIDIA GPU (memory>=14G) + CUDA cuDNN

# How to Train
1.Set the location of the dataset in the Config file
2.Prepare your dataset, which should include MRI and table datasets
```
|--Train
|----Subject1_id-PTID-label.nii.gz
|----Subject2_id-PTID-label.nii.gz
|--test
|----Subject1_id-PTID-label.nii.gz
|----Subject2_id-PTID-label.nii.gz
```
3.Download the pre-trained  [3D GAN-Vit](https://drive.google.com/drive/folders/1TMPE6JLMW87uMGIzYsbEZsgxmTArnlYE?usp=share_link) and [Mamba Classifier](https://drive.google.com/drive/folders/1AXTFHRLqQe1VKwscngRCjmuA-rxfETtB?usp=drive_link) parameters

4.specify the location of parameter file and use python test_mamba.py
