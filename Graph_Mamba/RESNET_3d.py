import torch
from torch import nn

class Bottleneck(nn.Module):  # 瓶颈模块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out  #


class ResNet_3d(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_3d, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)  # conv1的输出维度
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)  # H/2,W/2。C不变
        self.layer1 = self._make_layer(block, 64, layers[0])  # H,W不变。downsample控制的shortcut，out_channel=64x4=256
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2)  # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4)  # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048
        # 最后的分类层--------------------------------------------------------------------------------------
        # self.conv_cls = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Dropout(0.1),
        #                               nn.Linear(in_features=512 * block.expansion, out_features=2, bias=True))
        self.down_layer = nn.Conv3d(in_channels=2048, out_channels=512, kernel_size=1)
        self.down_layer2 = nn.Conv3d(in_channels=512, out_channels=108, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion  # 在下一次调用_make_layer函数的时候，self.in_channel已经x4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)  # '*'的作用是将list转换为非关键字参数传入

    def forward(self, x):
        x = self.conv1(x) #c,hwd  --> 64, hwd//2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #64, hwd//4
        x = self.layer1(x)  #256, hwd//4
        x = self.layer2(x)  #512, hwd//8
        x = self.layer3(x)  #1024, hwd//8
        x = self.layer4(x)  #2048, hwd//8
        x = self.down_layer(x)
        x = self.down_layer2(x)#108, hwd//8

        # x = self.conv_cls(x)
        return x

if __name__ == '__main__':
    x = torch.rand(2, 1, 160, 160,96)
    model = ResNet_3d(Bottleneck, [3, 4, 6, 3], )
    y = model(x)
    print(y.shape)