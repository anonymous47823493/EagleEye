import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # model params
        self.parser.add_argument('--model_name', type=str, default="mobilenetv1", help="what kind of model you are using. Only support `resnet50`, `mobilenetv1` and `mobilenetv1_imagenet`")
        self.parser.add_argument('--num_classes', type=int, default=1000, help="num of class label")
        self.parser.add_argument('--pruned_model', type=str, default="", help='path to pruned model state dict')

        # env params
        self.parser.add_argument('--gpu_ids', type=int, default=[0], nargs="+", help='GPU ids.')

        # fine-tune params
        self.parser.add_argument('--batch_size', type=int, default=64, help="batch size while fine-tuning")
        self.parser.add_argument('--dataset_path', type=str, default="./cifar10", help="path to dataset")
        self.parser.add_argument('--dataset_name', type=str, default="cifar10_224", help="filename of the file contains your own `get_dataloaders` function")
        self.parser.add_argument('--num_workers', type=int, default=2, help='Number of workers used in dataloading')
        self.initialized = True
    
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt