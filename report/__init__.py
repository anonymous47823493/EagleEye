import distiller
import torch
import os
import os.path as osp
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime


def weights_sparsity_summary(model, opt=None):
    try:
        df = distiller.weights_sparsity_summary(
            model.module, return_total_sparsity=True
        )
    except AttributeError:
        df = distiller.weights_sparsity_summary(model, return_total_sparsity=True)
    return df[0]["NNZ (dense)"].sum() // 2


def performance_summary(model, dummy_input, opt=None, prefix=""):
    try:
        df = distiller.model_performance_summary(model.module, dummy_input)
    except AttributeError:
        df = distiller.model_performance_summary(model, dummy_input)
    new_entry = {
        "Name": ["Total"],
        "MACs": [df["MACs"].sum()],
    }
    MAC_total = df["MACs"].sum()
    return MAC_total


def model_summary(model, dummy_input, opt=None):
    return (
        performance_summary(model, dummy_input, opt),
        weights_sparsity_summary(model, opt),
    )


def _check_mk_path(path):
    if not osp.exists(path):
        os.makedirs(path)


class Reporter:
    def __init__(self, opt, use_time=True):
        now = datetime.now().strftime("-%Y-%m-%d-%H:%M:%S")

        if use_time:
            self.log_dir = osp.join(opt.log_dir, opt.exp_name + now)
        else:
            self.log_dir = osp.join(opt.log_dir, opt.exp_name)

        _check_mk_path(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.ckpt_log_dir = osp.join(self.log_dir, "checkpoints")
        _check_mk_path(self.ckpt_log_dir)

        self.config_log_dir = osp.join(self.log_dir, "config")
        _check_mk_path(self.config_log_dir)

    def log_config(self, path):
        target = osp.join(self.config_log_dir, path.split("/")[-1])
        shutil.copyfile(path, target)

    def get_writer(self):
        return self.writer

    def log_metric(self, key, value, step):
        self.writer.add_scalar("data/" + key, value, step)

    def log_text(self, msg):
        print(msg)

    def save_checkpoint(self, state_dict, ckpt_name, epoch=0):
        checkpoint = {"state_dict": state_dict, "epoch": epoch}
        torch.save(checkpoint, osp.join(self.ckpt_log_dir, ckpt_name))
