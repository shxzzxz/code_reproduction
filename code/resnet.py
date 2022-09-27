from turtle import forward
import torch
from torch import nn

class BasicBlock(nn.Module):
    expandsion = 1
    def __init__(self,input_dim,output_dim,downsample) -> None:
        super(BasicBlock,self).__init__()
        if downsample:
            self.downsample = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2)
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)

class BottleNeckBlock(nn.Module):
    expandsion = 4
    def __init__(self) -> None:
        super(BottleNeckBlock,self).__init__()

class resnet(nn.Module):
    input_dim = 64
    def __init__(self,block) -> None:
        super(resnet,self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(3, stride=2)
        self.conv2_x = _make_layer(64)
        self.conv3_x = _make_layer(128)
        self.conv4_x = _make_layer(256)
        self.conv5_x = _make_layer(512)
    def _make_layer(self,layer_dim):
        if
        self.block(input_dim)
