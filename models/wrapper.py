#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import shutil
import torch.optim as optim
import numpy as np


class ModelWrapper(nn.Module):
    def __init__(self, opt):
        super(ModelWrapper, self).__init__()
        if opt.model_name == "mobilenetv1":
            from .mobilenet import MobileNet

            self._net = MobileNet(num_classes=opt.num_classes)
        elif opt.model_name == "resnet50":
            from .resnet import resnet50

            self._net = resnet50(num_classes=opt.num_classes)

        self.optimizer = optim.SGD(
            self._net.parameters(),
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, x):  # test forward
        x, _ = x

        self._net.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        out = self._net(x)

        return out

    def get_compress_part(self):
        return self._net

    def parallel(self, gpu_ids):
        if len(gpu_ids) > 1:
            self._net = nn.DataParallel(self._net)

    def get_loss(self, inputs):
        device = next(self.parameters()).device

        self._net.train()
        images, targets = inputs
        images, targets = images.to(device), targets.to(device)
        out = self._net(images)
        loss = self._criterion(out, targets)

        return loss

    def get_eval_scores(self, dataloader_test):
        from tqdm import tqdm

        device = next(self.parameters()).device
        to_cuda = next(self.parameters()).device.type == "cuda"

        total = 0
        correct = 0

        top_5_acc = 0

        self._net.eval()
        # print('==> evaluating accuracy')
        with torch.no_grad():
            for i, sample in enumerate(
                tqdm(dataloader_test, leave=False, desc="evaluating accuracy")
            ):
                outputs = self.forward(sample)
                _, predicted = outputs.max(1)
                targets = sample[1].to(device)

                prec5 = accuracy(outputs.data, targets.data, topk=(5,))
                prec5 = prec5[0]
                top_5_acc += prec5.item() * targets.size(0)

                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        top_5_acc /= total
        scores = {"accuracy": round(acc, 3)}

        return scores

    def load_checkpoint(self, checkpoint_file):
        """
        Function to load pruned model or normal model checkpoint.
        :param str checkpoint_file: path to checkpoint file, such as `models/ckpt/mobilenet.pth`
        """
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        net = self.get_compress_part()
        #### load pruned model ####
        for key, module in net.named_modules():
            # torch.nn.BatchNorm2d
            if isinstance(module, nn.BatchNorm2d):
                module.weight = torch.nn.Parameter(checkpoint[key + ".weight"])
                module.bias = torch.nn.Parameter(checkpoint[key + ".bias"])
                module.num_features = module.weight.size(0)
                module.running_mean = module.running_mean[0 : module.num_features]
                module.running_var = module.running_var[0 : module.num_features]
            # torch.nn.Conv2d
            elif isinstance(module, nn.Conv2d):
                # for conv2d layer, bias and groups should be consider
                module.weight = torch.nn.Parameter(checkpoint[key + ".weight"])
                module.out_channels = module.weight.size(0)
                module.in_channels = module.weight.size(1)
                if module.groups is not 1:
                    # group convolution case
                    # only support for MobileNet, pointwise conv
                    module.in_channels = module.weight.size(0)
                    module.groups = module.in_channels
                if key + ".bias" in checkpoint:
                    module.bias = torch.nn.Parameter(checkpoint[key + ".bias"])
            # torch.nn.Linear
            elif isinstance(module, nn.Linear):
                module.weight = torch.nn.Parameter(checkpoint[key + ".weight"])
                if key + ".bias" in checkpoint:
                    module.bias = torch.nn.Parameter(checkpoint[key + ".bias"])
                module.out_features = module.weight.size(0)
                module.in_features = module.weight.size(1)

        net.load_state_dict(checkpoint)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    num = output.size(1)
    target_topk = []
    appendices = []
    for k in topk:
        if k <= num:
            target_topk.append(k)
        else:
            appendices.append([0.0])
    topk = target_topk
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res + appendices
