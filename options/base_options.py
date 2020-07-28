#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import argparse
import os


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # model params
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="mobilenetv1",
            help="what kind of model you are using. Only support `resnet50`, `mobilenetv1` and `mobilenetv1_imagenet`",
        )
        self.parser.add_argument(
            "--num_classes", type=int, default=1000, help="num of class label"
        )
        self.parser.add_argument(
            "--checkpoint", type=str, default="", help="path to model state dict"
        )

        # env params
        self.parser.add_argument(
            "--gpu_ids", type=int, default=[0], nargs="+", help="GPU ids."
        )

        # fine-tune params
        self.parser.add_argument(
            "--batch_size", type=int, default=64, help="batch size while fine-tuning"
        )
        self.parser.add_argument(
            "--epoch", type=int, default=120, help="epoch while fine-tuning"
        )
        self.parser.add_argument(
            "--dataset_path", type=str, default="./cifar10", help="path to dataset"
        )
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default="cifar10_224",
            help="filename of the file contains your own `get_dataloaders` function",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=16,
            help="Number of workers used in dataloading",
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.01, help="learning rate while fine-tuning"
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=5e-4,
            help="weight decay while fine-tuning",
        )
        self.parser.add_argument(
            "--momentum", type=float, default=0.9, help="momentum while fine-tuning"
        )

        self.parser.add_argument(
            "--search_result",
            type=str,
            default="mbv1.txt",
            help="path to search result",
        )
        self.parser.add_argument(
            "--strategy_id", type=int, default=0, help="line num in search result file"
        )

        self.parser.add_argument("--log_dir", type=str, default="logs/", help="log dir")
        self.parser.add_argument(
            "--exp_name", type=str, default="mbv1_50flops", help="experiment name"
        )

        # search params
        self.parser.add_argument(
            "--max_rate", type=float, default=0.7, help="define search space"
        )
        self.parser.add_argument(
            "--min_rate", type=float, default=0, help="define search space"
        )
        self.parser.add_argument(
            "--compress_schedule_path",
            type=str,
            default="compress_config/mbv1_imagenet.yaml",
            help="path to compression schedule",
        )
        self.parser.add_argument(
            "--flops_target",
            type=float,
            default=0.5,
            help="flops constraints for pruning",
        )
        self.parser.add_argument(
            "--output_file", type=str, default="mbv1.txt", help="path to search result"
        )

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
