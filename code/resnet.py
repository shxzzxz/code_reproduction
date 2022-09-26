from tkinter.ttk import _Padding
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self) -> None:
        super(BasicBlock,self).__init__()

class BottleNeckBlock(nn.Module):
    expandsion = 4
    def __init__(self) -> None:
        super(BottleNeckBlock,self).__init__()

class resnet(nn.Module):
    def __init__(self) -> None:
        super(resnet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0)
