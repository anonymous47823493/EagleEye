# EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning

![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch version support](https://img.shields.io/badge/pytorch-1.1.0-red.svg)

PyTorch implementation for *[EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491)*

[Bailin Li,](https://github.com/bezorro) [Bowen Wu](https://github.com/Bowenwu1), Jiang Su, [Guangrun Wang](https://wanggrun.github.io/projects/zw), [Liang Lin](http://www.linliang.net/)

Presented at [ECCV 2020 (Oral)](https://eccv2020.eu/accepted-papers/)

Check [slides](https://dmmo.dm-ai.cn/eagle_eye/dmai_eagleeye_jiqizhixin202008.pdf) about EagleEye: “High-performance AI on the Edge: from perspectives of model compression and hardware architecture design“, DMAI HiPerAIR, Aug. 2020.

![pipeline](fig/eye.png)

## Citation

If you use EagleEye in your research, please consider citing:

```
@misc{li2020eagleeye,
    title={EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning},
    author={Bailin Li and Bowen Wu and Jiang Su and Guangrun Wang and Liang Lin},
    year={2020},
    eprint={2007.02491},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Adaptive-BN-based Candidate Evaluation

For the ease of your own implementation, here we present the key code for proposed Adaptive-BN-based Candidate Evaluation. The official implementation will be released soon.

```python
def eval_pruning_strategy(model, pruning_strategy, dataloader_train):
   # Apply filter pruning to trained model
   pruned_model = prune(model, pruning_strategy)

   # Adaptive-BN
   pruned_model.train()
   max_iter = 100
   with torch.no_grad():
      for iter_in_epoch, sample in enumerate(dataloader_train):
            pruned_model.forward(sample)
            if iter_in_epoch > max_iter:
                break

   # Eval top-1 accuracy for pruned model
   acc = pruned_model.get_val_acc()
   return acc
```

## Baseline Model Training

The code used for training baseline models(MobileNetV1, ResNet50) will be released at [CNNResearchToolkit](https://github.com/Bowenwu1/CNNResearchToolkit). Welcome everyone to follow!

## Setup

1. **Prepare Data**

   Download `ILSVRC2012` dataset from http://image-net.org/challenges/LSVRC/2012/index#introduction

2. **Download Pretrained Models**

   We provide pretrained baseline models and reported pruned models in  [Google Drive](<https://drive.google.com/drive/folders/1ENq4RuFey3J2iL-Lu1BZ9ToTYILpV9bC>). Please put the downloaded models in the dir of `models/ckpt/`.

3. **Prepare Runtime Environment**

   ```shell
   pip install -r requirements.txt
   ```

## Usage

Our proposed EagleEye contains 3 steps:

1. Adaptive-BN-based Searching for Pruning Strategy
2. Candidate Selection
3. Fine-tuning of Pruned Model

### 1. Adaptive-BN-based Searching for Pruning Strategy

On this step, pruning strategies are randomly generated. Then, Adaptive-BN-based evaluation are performed among these pruning strategies. Pruning strategies and their eval scores will be saved to `search_results/pruning_strategies.txt`.

If you do not want to perform searching by yourself, the provided search result could be found in `search_results/`.

Parameters involved in this steps:

|Name|Description|
|----|-----------|
|`--flops_target`|The remaining ratio of FLOPs of pruned model|
|`--max_rate`<br>`--min_rate`|Define the search space. The search space is [min_rate, max_rate]|
|`--output_file`|File stores the searching results.|

Sample scripts could refer to `1. Search` of `scripts/mbv1_50flops.sh`.

**Searching space for different models**

|Model|Pruned FLOPs|[min_rate, max_rate]|
|-----|-----|--------------------|
|MobileNetV1|-50%|[0, 0.7]|
|ResNet50|-25%|[0, 0.4]|
|ResNet50|-50%|[0, 0.7]|
|ResNet50|-75%|[0, 0.8]|

### 2. Candidate Selection

On this step, best pruning strategy is picked from `output_file` generated on step1.

The output looks like as following:
```
########## pruning_strategies.txt ##########
strategy index:84, score:0.143
strategy index:985, score:0.123
```

Sample scripts could refer to `2. Selection` of `scripts/mbv1_50flops.sh`.

### 3. Fine-tuning of Pruned Model

This step take strategy index as input and perform fine-tuning on it.

Parameters involved in this steps:

|Name|Description|
|----|-----------|
|`--search_result`|Searching results|
|`--strategy_id`|Index of best pruning strategy from step2|
|`--lr`|Learning rate for fine-tuning|
|`--weight_decay`|Weight decay while fine-tuning|
|`--epoch`|Number of fine-tuning epoch|

Sample scripts could refer to `3. Fine-tuning` of `scripts/mbv1_50flops.sh`.



## Inference of Pruned Model

**For ResNet50:**

```shell
python3 inference.py \
--model_name resnet50 \
--num_classes 1000 \
--checkpoint models/ckpt/{resnet50_25flops.pth|resnet50_50flops.pth|resnet50_72flops.pth} \
--gpu_ids 4 \
--batch_size 512 \
--dataset_path {PATH_TO_IMAGENET} \
--dataset_name imagenet \
--num_workers 20
```

**For MobileNetV1:**

```shell
python3 inference.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/mobilenetv1_50flops.pth \
--gpu_ids 4 \
--batch_size 512 \
--dataset_path {PATH_TO_IMAGENET} \
--dataset_name imagenet \
--num_workers 20
```

After running above program, the output looks like below:

```
######### Report #########                                                                                                                                                  
Model:resnet50
Checkpoint:models/ckpt/resnet50_50flops_7637.pth
FLOPs of Original Model:4.089G;Params of Original Model:25.50M
FLOPs of Pruned   Model:2.057G;Params of Pruned   Model:14.37M
Top-1 Acc of Pruned Model on imagenet:0.76366
##########################
```


## Results

### Quantitative analysis of correlation

Correlation between evaluation and fine-tuning accuracy with different pruning ratios (MobileNet V1 on ImageNet classification Top-1 results)

![corr](fig/cor_fix_flops.png)

### Results on ImageNet

| Model | FLOPs | Top-1 Acc | Top-5 Acc | Checkpoint |
| ---   | ----  |  -------  | --------  | ---------------- |
| ResNet-50 | 3G<br>2G<br>1G | 77.1%<br>76.4%<br>74.2%| 93.37%<br>92.89%<br>91.77% | [resnet50_75flops.pth](https://drive.google.com/file/d/1oPQOZJdKwZPXPSLykLkruHAFhxFmdzHp/view?usp=sharing) <br> [resnet50_50flops.pth](https://drive.google.com/file/d/19eOUO0LTzrQ-9izO4OzXcg83XAGwxf7u/view?usp=sharing) <br> [resnet50_25flops.pth](https://drive.google.com/file/d/1ppBLtajt5xcwa5xoonwn1T98MTB0V9DU/view?usp=sharing) |
| MobileNetV1 | 284M | 70.9% |  89.62% | [mobilenetv1_50flops.pth](https://drive.google.com/file/d/1LZGqk_oPXNYcGa5Gk93fmHxRgPdfzf9p/view?usp=sharing) |

### Results on CIFAR-10

| Model | FLOPs | Top-1 Acc |
| ---   | ----  |  -----    |
| ResNet-50 | 62.23M | 94.66% |
| MobileNetV1 | 26.5M<br>12.1M<br>3.3M | 91.89% <br> 91.44% <br> 88.01% |

