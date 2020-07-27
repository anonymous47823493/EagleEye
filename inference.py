#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os
import torch
from options.base_options import BaseOptions
from models.wrapper import ModelWrapper
from report import model_summary
from data import custom_get_dataloaders
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np


def main():
    # get options
    opt = BaseOptions().parse()
    # basic settings
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    ##################### Get Dataloader ####################
    _, dataloader_test = custom_get_dataloaders(opt)
    # dummy_input is sample input of dataloaders
    if hasattr(dataloader_test, "dataset"):
        dummy_input = dataloader_test.dataset.__getitem__(0)
        dummy_input = dummy_input[0]
        dummy_input = dummy_input.unsqueeze(0)
    else:
        # for imagenet dali loader
        dummy_input = torch.rand(1, 3, 224, 224)
    #####################  Evaluate Baseline Model  ####################
    net = ModelWrapper(opt)
    net = net.to(device)
    net.parallel(opt.gpu_ids)
    flops_before, params_before = model_summary(net.get_compress_part(), dummy_input)

    del net
    #####################  Evaluate Pruned Model  ####################
    net = ModelWrapper(opt)
    net.load_checkpoint(opt.checkpoint)
    net = net.to(device)
    flops_after, params_after = model_summary(net.get_compress_part(), dummy_input)
    net.parallel(opt.gpu_ids)

    acc_after = net.get_eval_scores(dataloader_test)

    #################### Report #####################
    print("######### Report #########")
    print("Model:{}".format(opt.model_name))
    print("Checkpoint:{}".format(opt.checkpoint))
    print(
        "FLOPs of Original Model:{:.3f}G;Params of Original Model:{:.2f}M".format(
            flops_before / 1e9, params_before / 1e6
        )
    )
    print(
        "FLOPs of Pruned   Model:{:.3f}G;Params of Pruned   Model:{:.2f}M".format(
            flops_after / 1e9, params_after / 1e6
        )
    )
    print(
        "Top-1 Acc of Pruned Model on {}:{}".format(
            opt.dataset_name, acc_after["accuracy"]
        )
    )
    print("##########################")


if __name__ == "__main__":
    main()
