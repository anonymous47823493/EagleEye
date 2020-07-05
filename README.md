# EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning



![pipeline](fig/eye.png)


## Results

### Quantitative analysis of correlation

Correlation between evaluation and fine-tuning accuracy with different pruning ratios (MobileNet V1 on ImageNet classification Top-1 results)

![corr](fig/cor_fix_flops.png)

### Results on ImageNet

| Model | FLOPs | Top-1 Acc |
| ---   | ----  |  -------  |
| ResNet-50 | 3G<br>2G<br>1G | 77.1%<br>76.4%<br>74.2%|
| MobileNetV1 | 284M | 70.9% |

### Results on CIFAR-10

| Model | FLOPs | Top-1 Acc |
| ---   | ----  |  -----    |
| ResNet-56 | 62.23M | 94.66% |
| MobileNetV1 | 26.5M<br>12.1M<br>3.3M | 91.89% <br> 91.44% <br> 88.01% |
## Setup

1. **Prepare Data**

   Download `ILSVRC2012` dataset from http://image-net.org/challenges/LSVRC/2012/index#introduction

2. **Download Pretrained Models**

   We provide reported pruned models in [Dropbox](<https://www.dropbox.com/sh/im1janxv5p8u5jm/AAA7s6danrqdL42UvteICARra?dl=0>). Please put the downloaded models in the dir of `models/ckpt/`.

3. **Prepare Runtime Environment**

   ```shell
   pip install -r requirements.txt
   ```

## Usage

```shell
usage: main.py [-h] [--model_name MODEL_NAME] [--num_classes NUM_CLASSES]
               [--pruned_model PRUNED_MODEL] [--gpu_ids GPU_IDS [GPU_IDS ...]]
               [--batch_size BATCH_SIZE] [--dataset_path DATASET_PATH]
               [--dataset_name DATASET_NAME] [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        what kind of model you are using. Only support
                        `resnet50` and `mobilenetv1`.
  --num_classes NUM_CLASSES
                        num of class label
  --pruned_model PRUNED_MODEL
                        path to pruned model state dict
  --gpu_ids GPU_IDS [GPU_IDS ...]
                        GPU ids.
  --batch_size BATCH_SIZE
                        batch size while fine-tuning
  --dataset_path DATASET_PATH
                        path to dataset
  --dataset_name DATASET_NAME
                        filename of the file contains your own
                        `get_dataloaders` function
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
```



**For ResNet50:**

```shell
python3 main.py \
--model_name resnet50 \
--num_classes 1000 \
--pruned_model models/ckpt/{resnet50_25flops.pth|resnet50_50flops.pth|resnet50_72flops.pth} \
--gpu_ids 4 \
--batch_size 512 \
--dataset_path {PATH_TO_IMAGENET} \
--dataset_name imagenet \
--num_workers 20
```

**For MobileNetV1:**

```shell
python3 main.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--pruned_model models/ckpt/mobilenetv1_50flops.pth \
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

