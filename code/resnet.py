import torch
from torch import nn

class BasicBlock(nn.Module):
    expandsion = 1
    def __init__(self) -> None:
        super(BasicBlock,self).__init__()

class BottleNeckBlock(nn.Module):
    expandsion = 4
    def __init__(self) -> None:
        super(BottleNeckBlock,self).__init__()

class resnet(nn.Module):
    input_dim = 64
    def __init__(self,block) -> None:
        super(resnet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(3, stride=2)
        self.conv2_x = _make_layer(block,downsample=False)
        self.conv3_x = _make_layer(block,downsample=False)
        self.conv4_x = _make_layer(block,downsample=False)
        self.conv5_x = _make_layer(block,downsample=False)
    def _make_layer(self,block,downsample):
        if downsample:
            self.conv1 = nn.Conv2d(in_channels=self.input_dim, out_channels=64, kernel_size=7, stride=2, padding=0)
            identity = 

