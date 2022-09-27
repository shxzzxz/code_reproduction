import torch
from torch import nn

class BasicBlock(nn.Module):
    expandsion = 1
    def __init__(self,input_dim, layer_dim, stride, **kwargs) -> None:
        super(BasicBlock,self).__init__()
        output_dim = layer_dim*self.expandsion
        if input_dim != output_dim or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_dim))
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_dim)

        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_dim)
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = self.relu(x)

class BottleNeckBlock(nn.Module):
    expandsion = 4
    def __init__(self, input_dim, layer_dim, stride, groups=1, width_per_group=64) -> None:
        super(BottleNeckBlock,self).__init__()
        output_dim = layer_dim*self.expandsion

        width = int(layer_dim * (width_per_group / 64.)) * groups    # 此处乘子未进行验证，直接copy得到

        if input_dim != output_dim or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_dim))
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, 
                                stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=output_dim, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_dim)
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + identity
        x = self.relu(x)
class ResNet(nn.Module):

    def __init__(self,block,block_nums,num_classes=1000, groups=1,
                 width_per_group=64) -> None:
        super(ResNet,self).__init__()
        self.groups = groups
        self.width_per_group = width_per_group

        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpooling = nn.MaxPool2d(3, stride=2)

        self.conv2 = self._make_layer(block, 64, block_nums[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, block_nums[1])
        self.conv4_x = self._make_layer(block, 256, block_nums[2])
        self.conv5_x = self._make_layer(block, 512, block_nums[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expandsion, num_classes)

    def _make_layer(self,block,layer_dim,block_num,stride=2):

        layer_list = []

        layer_list.append(block(self.in_channel, layer_dim, stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = block.expandsion * layer_dim

        for _ in range(1,block_num):
            layer_list.append(block(self.in_channel, layer_dim, stride=1, groups=self.groups,
                            width_per_group=self.width_per_group))
        
        return  nn.Sequential(*layer_list)
    
    def forward(self):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.maxpooling(x)
        x = self.conv2(x)

        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = self.fc(x)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes)

def resnet50(num_classes=1000):
    return ResNet(BottleNeckBlock,[3,4,6,3],num_classes=num_classes)


def resnet101(num_classes=1000):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(BottleNeckBlock, [3, 4, 23, 3], num_classes=num_classes)


def resnext50_32x4d(num_classes=1000):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(BottleNeckBlock, [3, 4, 6, 3],
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(BottleNeckBlock, [3, 4, 23, 3],
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=width_per_group)

if __name__ == "__main__":
    net = resnext50_32x4d()
    print(net)

