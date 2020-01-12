from torch import nn
from collections import OrderedDict


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=True)
        self.sigmoid = nn.Sigmoid()
        # self.fc = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(channel, channel // reduction, bias=True)),
        #     ('relu', nn.ReLU(inplace=True)),
        #     ('fc2', nn.Linear(channel // reduction, channel, bias=True)),
        #     ('sigmoid', nn.Sigmoid())]
        # ))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
