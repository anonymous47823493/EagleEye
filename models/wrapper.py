import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import shutil
import torch.optim as optim


class ModelWrapper(nn.Module):
    def __init__(self, opt):
        super(ModelWrapper, self).__init__()
        if opt.model_name == "mobilenetv1":
            from .mobilenet import MobileNet
            self._net = MobileNet(num_classes=opt.num_classes)
        elif opt.model_name == "resnet50":
            from .resnet import resnet50
            self._net = resnet50(num_classes=opt.num_classes)

        self._criterion = nn.CrossEntropyLoss()
    
    def forward(self, x): # test forward
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
        to_cuda = next(self.parameters()).device.type == 'cuda'

        total = 0
        correct = 0

        self._net.eval()
        # print('==> evaluating accuracy')
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataloader_test, leave=False, desc='evaluating accuracy')):
                outputs = self.forward(sample)
                _, predicted = outputs.max(1)
                targets = sample[1].to(device)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        acc = correct / total

        scores = {'accuracy' : round(acc,1)}

        return scores
    
    def load_checkpoint(self, checkpoint_file):
        """
        Function to load pruned model or normal model checkpoint.
        :param str checkpoint_file: path to checkpoint file, such as `models/ckpt/mobilenet.pth`
        """
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        net = self.get_compress_part()
        #### load pruned model ####
        for key, module in net.named_modules():
            # torch.nn.BatchNorm2d
            if isinstance(module, nn.BatchNorm2d):
                module.weight       = torch.nn.Parameter(checkpoint[key + '.weight'])
                module.bias         = torch.nn.Parameter(checkpoint[key + '.bias'])
                module.num_features = module.weight.size(0)
                module.running_mean = module.running_mean[0:module.num_features]
                module.running_var  = module.running_var[0:module.num_features]
            # torch.nn.Conv2d
            elif isinstance(module, nn.Conv2d):
                # for conv2d layer, bias and groups should be consider
                module.weight       = torch.nn.Parameter(checkpoint[key + '.weight'])
                module.out_channels = module.weight.size(0)
                module.in_channels = module.weight.size(1)
                if module.groups is not 1:
                    # group convolution case
                    # only support for MobileNet, pointwise conv
                    module.in_channels  = module.weight.size(0) 
                    module.groups = module.in_channels
                if key + '.bias' in checkpoint:
                    module.bias   = torch.nn.Parameter(checkpoint[key + '.bias'])
            # torch.nn.Linear
            elif isinstance(module, nn.Linear):
                module.weight       = torch.nn.Parameter(checkpoint[key + '.weight'])
                if key + '.bias' in checkpoint:
                    module.bias = torch.nn.Parameter(checkpoint[key + '.bias'])
                module.out_features = module.weight.size(0)
                module.in_features  = module.weight.size(1)

        net.load_state_dict(checkpoint)
