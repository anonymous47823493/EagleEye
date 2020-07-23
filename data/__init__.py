#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os
import torch
import importlib


def custom_get_dataloaders(opt):
    dataset_filename = "data." + opt.dataset_name
    datasetlib = importlib.import_module(dataset_filename)
    # find method named `get_dataloaders`
    for name, method in datasetlib.__dict__.items():
        if name.lower() == "get_dataloaders":
            get_data_func = method
    return get_data_func(opt.batch_size, opt.num_workers, path=opt.dataset_path)
