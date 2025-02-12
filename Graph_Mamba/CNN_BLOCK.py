import torch
from torch import nn


class CNN_branch(nn.Module):
    def __init__(self, ):
        super().__init__()
        sequence1 = [
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 3, 3), stride=2, dilation=1, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU()
        ]
        sequence2 = [
            nn.Conv3d(in_channels=4, out_channels=16, kernel_size=(3, 3, 3), stride=2, dilation=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        ]
        sequence3 = [
            nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(3, 3, 3), stride=2, dilation=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        ]

        self.conv1 = nn.Sequential(*sequence1)
        self.conv2 = nn.Sequential(*sequence2)
        self.conv3 = nn.Sequential(*sequence3)

    def forward(self, x, ):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)

        return y1, y2, y3


class CNN_branch2(nn.Module):
    def __init__(self, ):
        super().__init__()

        sequence2 = [
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=2, dilation=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        ]
        sequence3 = [
            nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(3, 3, 3), stride=2, dilation=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        ]

        self.conv2 = nn.Sequential(*sequence2)
        self.conv3 = nn.Sequential(*sequence3)

    def forward(self, x, ):
        y2 = self.conv2(x)
        y3 = self.conv3(y2)

        return y3


class CNN_branch3(nn.Module):
    def __init__(self, ):
        super().__init__()

        sequence3 = [
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=2, dilation=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        ]

        self.conv3 = nn.Sequential(*sequence3)

    def forward(self, x, ):
        y3 = self.conv3(x)

        return y3


class CNN_Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN_a = CNN_branch()
        self.CNN_b = CNN_branch()

        self.branch1 = CNN_branch2()

        self.branch2 = CNN_branch3()

        self.fusion_layer = nn.Conv3d(in_channels=64 * 4, out_channels=64, kernel_size=1)

    def forward(self, MRI, PET):
        x1, x2, x3 = self.CNN_a(MRI)
        y1, y2, y3 = self.CNN_b(PET)

        branch1 = self.branch1(torch.cat([x1, y1], dim=1))
        branch2 = self.branch2(torch.cat([x2, y2], dim=1))

        out = self.fusion_layer(torch.cat([x3, y3, branch1, branch2], dim=1))
        return out


if __name__ == '__main__':
    MRI = torch.rand(1, 1, 160, 160, 96)
    PET = torch.rand(1, 1, 160, 160, 96)
    model = CNN_Fusion()
    y = model(MRI, PET)
    print(y.shape)
