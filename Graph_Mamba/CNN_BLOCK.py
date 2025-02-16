import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

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


class CNN_Fusion2(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN_a = CNN_branch()
        self.CNN_b = CNN_branch()

        # self.branch1 = CNN_branch2()
        #
        # self.branch2 = CNN_branch3()

        # self.fusion_layer = nn.Conv3d(in_channels=64 * 4, out_channels=64, kernel_size=1)

    def forward(self, MRI, PET):
        x1, x2, x3 = self.CNN_a(MRI)
        y1, y2, y3 = self.CNN_b(PET)

        # branch1 = self.branch1(torch.cat([x1, y1], dim=1))
        # branch2 = self.branch2(torch.cat([x2, y2], dim=1))

        # out = torch.cat([x3, y3, branch1, branch2], dim=1)

        return x3, y3


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class PyramidLayer(nn.Module):
    """Mamba结构+金字塔池化的混合层，用于多尺度特征建模"""

    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        """
        Args:
            in_chs: 输入通道数
            dim: 中间特征维度
            d_state: Mamba的状态空间维度
            d_conv: Mamba的局部卷积宽度
            expand: Mamba块扩展因子
            last_feat_size: 特征图最终尺寸（用于生成池化比例）
        """
        super().__init__()

        # 生成池化比例序列（示例假设生成如[1,5,9,13,16]）
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)

        # 构建金字塔池化层组
        self.pool_layers = nn.ModuleList()

        # 第一个池化层：通道压缩+全局平均池化
        self.pool_layers.append(nn.Sequential(
            ConvBNReLU(in_chs, dim, kernel_size=1),  # 1x1卷积降维
            nn.AdaptiveAvgPool2d(1)  # 全局上下文信息
        ))

        # 后续池化层：多尺度特征提取
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),  # 多尺度池化
                    ConvBNReLU(in_chs, dim, kernel_size=1)  # 保持特征维度一致
                ))

        # Mamba序列建模模块（输入维度=原始通道+各池化层输出）
        # self.mamba = Mamba(
        #     d_model=dim * self.pool_len + in_chs,  # 总输入维度
        #     d_state=d_state,  # 状态空间维度
        #     d_conv=d_conv,  # 局部卷积核宽度
        #     expand=expand  # 通道扩展倍数
        # )

    def forward(self, x):
        res = x  # 残差连接
        B, C, H, W = res.shape

        # 多尺度特征收集（初始包含原始特征）
        ppm_out = [res]

        # 处理每个池化层
        for p in self.pool_layers:
            pool_out = p(x)
            # 上采样到原尺寸并收集
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)

        # 通道维度拼接 (B, in_chs + dim*pool_len, H, W)
        x = torch.cat(ppm_out, dim=1)
        # _, chs, _, _ = x.shape

        # 转换为序列格式 (B, H*W, C) 适合Mamba处理
        # x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)

        # Mamba序列建模
        # x = self.mamba(x)

        # 恢复空间维度 (B, C, H, W)
        # x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


class ConvBNReLU_3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm3d,
                 bias=False):
        super(ConvBNReLU_3d, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class PyramidLayer_3d(nn.Module):
    """Mamba结构+金字塔池化的混合层，用于多尺度特征建模"""

    def __init__(self, in_chs=512, dim=128, last_feat_size=16):
        """
        Args:
            in_chs: 输入通道数
            dim: 中间特征维度
            d_state: Mamba的状态空间维度
            d_conv: Mamba的局部卷积宽度
            expand: Mamba块扩展因子
            last_feat_size: 特征图最终尺寸（用于生成池化比例）
        """
        super().__init__()

        # 生成池化比例序列（示例假设生成如[1,5,9,13,]）
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)

        # 构建金字塔池化层组
        self.pool_layers = nn.ModuleList()

        # 第一个池化层：通道压缩+全局平均池化
        self.pool_layers.append(nn.Sequential(
            ConvBNReLU_3d(in_chs, dim, kernel_size=1),  # 1x1卷积降维
            nn.AdaptiveAvgPool3d(1)  # 全局上下文信息
        ))

        # 后续池化层：多尺度特征提取
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(pool_scale),  # 多尺度池化
                    ConvBNReLU_3d(in_chs, dim, kernel_size=1)  # 保持特征维度一致
                ))

    def forward(self, x):
        res = x  # 残差连接
        B, C, D, H, W = res.shape

        # 多尺度特征收集（初始包含原始特征）
        ppm_out = [res]

        # 处理每个池化层
        for p in self.pool_layers:
            pool_out = p(x)
            # 上采样到原尺寸并收集
            pool_out = F.interpolate(pool_out, (D, H, W), mode='trilinear', align_corners=False)
            ppm_out.append(pool_out)

        # 通道维度拼接 (B, in_chs + dim*pool_len, D, H, W)
        x = torch.cat(ppm_out, dim=1)

        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


if __name__ == '__main__':
    MRI = torch.rand(1, 1, 160, 160, 96)
    PET = torch.rand(1, 1, 160, 160, 96)
    model = CNN_Fusion()
    y = model(MRI, PET)
    print(y.shape)
