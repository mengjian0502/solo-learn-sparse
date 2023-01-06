import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, alpha:float=1.0):
        super(Net, self).__init__()
        self.alpha = alpha
        self.last_channel = int(1024*alpha)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32*alpha), 2), 
            conv_dw(int(32*alpha),  int(64*alpha), 1),
            conv_dw(int(64*alpha), int(128*alpha), 2),
            conv_dw(int(128*alpha), int(128*alpha), 1),
            conv_dw(int(128*alpha), int(256*alpha), 2),
            conv_dw(int(256*alpha), int(256*alpha), 1),
            conv_dw(int(256*alpha), int(512*alpha), 2),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(1024*alpha), 2),
            conv_dw(int(1024*alpha), int(1024*alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024*alpha), 100)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x

def mobilenet_v1(method, *args, **kwargs):
    model = Net(alpha=1)
    return model

def mobilenetv1_2x(method, *args, **kwargs):
    model = Net(alpha=1.25)
    return model