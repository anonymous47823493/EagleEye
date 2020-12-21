#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os
import torch
import torch.optim as optim
from options.base_options import BaseOptions
from models.wrapper import ModelWrapper
from report import model_summary, Reporter
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


def get_channel_config(path, line_num):
    # line_num starts from 0
    with open(path) as data:
        lines = data.readlines()
        i = 0
        for l in lines:
            if i == line_num:
                d = l.strip().split(" ")
                channel_config = []
                print("=" * 20, " read config")
                for i in range(0, 2):
                    print("{} ".format(d[i]), end="")
                for i in range(2, len(d)):
                    channel_config.append(float(d[i]))
                break
            i += 1
    return channel_config


def train_epoch(model_wrapper, dataloader_train, optimizer):
    optimizer.zero_grad()
    model_wrapper._net.train()

    loss_total = 0
    total = 0

    for iter_in_epoch, sample in enumerate(tqdm(dataloader_train, leave=False)):
        loss = model_wrapper.get_loss(sample)

        loss_total += loss.item()
        total += 1

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return loss_total / total


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

    #####################  Load Pruning Strategy ###############
    compression_scheduler = distiller.file_config(
        net.get_compress_part(), net.optimizer, opt.compress_schedule_path
    )

    channel_config = get_channel_config(
        opt.search_result, opt.strategy_id
    )  # pruning strategy

    compression_scheduler = random_compression_scheduler(
        compression_scheduler, channel_config
    )

    ###### Adaptive-BN-based Candidate Evaluation of Pruning Strategy ###
    thinning(net, compression_scheduler, input_tensor=dummy_input)

    flops_after, params_after = model_summary(net.get_compress_part(), dummy_input)
    ratio = flops_after / flops_before
    print("FLOPs ratio:", ratio)
    net = net.to(device)
    net.parallel(opt.gpu_ids)
    net.get_compress_part().train()
    with torch.no_grad():
        for index, sample in enumerate(tqdm(dataloader_train, leave=False)):
            _ = net.get_loss(sample)
            if index > 100:
                break

    strategy_score = net.get_eval_scores(dataloader_val)["accuracy"]

    print(
        "Result file:{}, Strategy ID:{}, Evaluation score:{}".format(
            opt.search_result, opt.strategy_id, strategy_score
        )
    )

    ##################### Fine-tuning #########################
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(net.optimizer, opt.epoch)
    reporter = Reporter(opt)
    best_acc = 0
    net._net.train()
    for epoch in range(1, opt.epoch + 1):
        reporter.log_metric("lr", net.optimizer.param_groups[0]["lr"], epoch)
        train_loss = train_epoch(net, dataloader_train, net.optimizer,)
        reporter.log_metric("train_loss", train_loss, epoch)

        lr_scheduler.step()

        scores = net.get_eval_scores(dataloader_val)
        print("==> Evaluation: Epoch={} Acc={}".format(epoch, str(scores)))

        reporter.log_metric("eval_acc", scores["accuracy"], epoch)

        if scores["accuracy"] > best_acc:
            best_acc = scores["accuracy"]
        reporter.log_metric("best_acc", best_acc, epoch)

        save_checkpoints(
            scores["accuracy"], net._net, reporter, opt.exp_name, epoch,
        )

        print("==> Training epoch %d" % epoch)


def save_checkpoints(acc, model, reporter, exp_name, epoch):
    if not hasattr(save_checkpoints, "best_acc"):
        save_checkpoints.best_acc = 0

    state_dict = model.state_dict()
    reporter.save_checkpoint(state_dict, "{}_latest.pth".format(exp_name), epoch)
    if acc > save_checkpoints.best_acc:
        reporter.save_checkpoint(state_dict, "{}_best.pth".format(exp_name), epoch)
        save_checkpoints.best_acc = acc
    reporter.save_checkpoint(state_dict, "{}_{}.pth".format(exp_name, epoch), epoch)


if __name__ == "__main__":
    # get options
    opt = BaseOptions().parse()
    main(opt)
