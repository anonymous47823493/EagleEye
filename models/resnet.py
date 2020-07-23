import torch.nn as nn
import torch
import math

__all__ = ["resnet50"]


def resnet50(**kwargs):
    import torchvision

    model = torchvision.models.resnet50(**kwargs)
    return model


if __name__ == "__main__":
    net = resnet50()
    image = torch.randn(2, 3, 224, 224)
    print(net)
    print(net.layer1[1].conv2)
    out = net(image)
    print(out.size())

    # print(distiller.weights_sparsity_summary(net))
