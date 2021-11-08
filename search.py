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
import distiller
from thinning import thinning


def random_compression_scheduler(compression_scheduler, channel_configuration):
    for i, item in enumerate(channel_configuration):
        compression_scheduler.policies[1][i].pruner.desired_sparsity = item
    return compression_scheduler


def get_pruning_strategy(opt, num_layer):
    channel_config = np.random.rand(num_layer)
    channel_config = channel_config * opt.max_rate
    channel_config = channel_config + opt.min_rate
    return channel_config


def main(opt):
    # basic settings
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    ##################### Get Dataloader ####################
    dataloader_train, dataloader_val = custom_get_dataloaders(opt)
    # dummy_input is sample input of dataloaders
    if hasattr(dataloader_val, "dataset"):
        dummy_input = dataloader_val.dataset.__getitem__(0)
        dummy_input = dummy_input[0]
        dummy_input = dummy_input.unsqueeze(0)
    else:
        # for imagenet dali loader
        dummy_input = torch.rand(1, 3, 224, 224)

    #####################  Create Baseline Model  ####################
    net = ModelWrapper(opt)
    net.load_checkpoint(opt.checkpoint)
    flops_before, params_before = model_summary(net.get_compress_part(), dummy_input)

    #####################  Pruning Strategy Generation ###############
    compression_scheduler = distiller.file_config(
        net.get_compress_part(), net.optimizer, opt.compress_schedule_path
    )
    num_layer = len(compression_scheduler.policies[1])

    channel_config = get_pruning_strategy(opt, num_layer)  # pruning strategy

    compression_scheduler = random_compression_scheduler(
        compression_scheduler, channel_config
    )

    ###### Adaptive-BN-based Candidate Evaluation of Pruning Strategy ###
    try:
        thinning(net, compression_scheduler, input_tensor=dummy_input)
    except Exception as e:
        print('[WARNING] This pruning strategy is invalid for distiller thinning module, pass it.')
        print(e)
        return

    flops_after, params_after = model_summary(net.get_compress_part(), dummy_input)
    ratio = flops_after / flops_before
    print("FLOPs ratio:", ratio)
    if ratio < opt.flops_target - 0.005 or ratio > opt.flops_target + 0.005:
        # illegal pruning strategy
        return
    net = net.to(device)
    net.parallel(opt.gpu_ids)
    net.get_compress_part().train()
    with torch.no_grad():
        for index, sample in enumerate(tqdm(dataloader_train, leave=False)):
            _ = net.get_loss(sample)
            if index > 100:
                break

    strategy_score = net.get_eval_scores(dataloader_val)["accuracy"]

    #################### Save Pruning Strategy and Score #########
    log_file = open(opt.output_file, "a+")
    log_file.write("{} {} ".format(strategy_score, ratio))

    for item in channel_config:
        log_file.write("{} ".format(str(item)))
    log_file.write("\n")
    log_file.close()
    print("Eval Score:{}".format(strategy_score))


if __name__ == "__main__":
    # get options
    opt = BaseOptions().parse()
    while True:
        main(opt)
