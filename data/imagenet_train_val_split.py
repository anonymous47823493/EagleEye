#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

np.random.seed(2019)
import random

random.seed(2019)
from torch.utils.data.sampler import SubsetRandomSampler
import pickle


def get_dataloaders(batch_size, n_workers, path=""):
    print("USE PART OF TRAIN SET WITH UNIFORM SPLIT")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = datasets.ImageFolder(
        osp.join(path, "train"),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        osp.join(path, "val"),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_idx_filename = "data/imagenet_train_val_split_idx.pickle"
    print("len(train_dataset)", len(train_dataset))
    if not osp.exists(val_idx_filename):
        val_size = 10000
        val_idx = []
        cls_start, cls_end = 0, 0
        for c_id in range(1000):
            for i in range(cls_start, len(train_dataset)):
                if train_dataset[i][1] == c_id:
                    cls_end = i + 1
                else:
                    break
            c_list = list(range(cls_start, cls_end))
            print("cid:{}, c_start:{}, c_end:{}".format(c_id, cls_start, cls_end))
            print(int(val_size / 1000))
            c_sample = random.sample(c_list, int(val_size / 1000))
            val_idx += c_sample
            cls_start = cls_end
        print("len of val_size:{}".format(len(val_idx)))
        pickle.dump(val_idx, open(val_idx_filename, "wb"))
    else:
        val_idx = pickle.load(open(val_idx_filename, "rb"))
    val_sampler = SubsetRandomSampler(val_idx)
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    return dataloader_train, dataloader_test
