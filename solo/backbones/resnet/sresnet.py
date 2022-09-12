import torch.nn as nn
import torch.nn.functional as F
from .sparsemodule import SparsConv2d

__all__ = ['sresnet18_imagenet', 'sresnet50_imagenet']

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, nspars=1, uspars=True):
        super(BasicBlock, self).__init__()

        self.conv1 = SparsConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = SparsConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SparsConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, nspars=nspars, uspars=uspars),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, nspars=1, uspars=True):
        super(Bottleneck, self).__init__()
        self.conv1 = SparsConv2d(in_planes, planes, kernel_size=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SparsConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SparsConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SparsConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, nspars=nspars, uspars=uspars),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, nspars=1, uspars=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.nspars = nspars
        self.uspars = uspars

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, SparsConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, self.nspars, self.uspars))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu0(self.bn1(out))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def sresnet18_imagenet(method, *args, **kwargs):
    return ResNet(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=1000, nspars=1, uspars=True)

def sresnet50_imagenet(method, *args, **kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3,4,6,3], num_classes=1000, nspars=1, uspars=True)