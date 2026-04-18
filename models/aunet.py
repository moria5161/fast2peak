import torch
from torch import nn
from abc import ABC
import torch.nn.functional as F

from . import register_model


class ChannelAttention1D(nn.Module, ABC):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.MaxPool1d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention1D(nn.Module, ABC):
    def __init__(self):
        super(SpatialAttention1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], 1)
        out = self.sigmoid(self.conv1d(out))
        return out


class CBAM(nn.Module, ABC):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention1D(channel)
        self.spatial_attention = SpatialAttention1D()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module, ABC):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(),
            nn.Conv1d(in_channels=places, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(),
            nn.Conv1d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places * self.expansion)
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(places * self.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    

class Conv_block_CA(nn.Module, ABC):
    def __init__(self, ch_in, ch_out):
        super(Conv_block_CA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )
        self.cbam = CBAM(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


class Up_conv_CA(nn.Module, ABC):
    def __init__(self, ch_in, ch_out):
        super(Up_conv_CA, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )
        self.cbam = CBAM(ch_out)

    def forward(self, x):
        x = self.up(x)
        x = self.cbam(x)
        return x


@register_model
class AUNet(nn.Module, ABC):
    def __init__(self, ch_in=1, ch_out=1, **kwargs):
        super(AUNet, self).__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block_CA(ch_in=ch_in, ch_out=64)
        self.Conv2 = Conv_block_CA(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block_CA(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block_CA(ch_in=256, ch_out=256)

        self.Up3 = Up_conv_CA(ch_in=256, ch_out=128)
        self.Up_conv3 = Conv_block_CA(ch_in=384, ch_out=128)

        self.Up2 = Up_conv_CA(ch_in=128, ch_out=64)
        self.Up_conv2 = Conv_block_CA(ch_in=192, ch_out=64)

        self.Up1 = Up_conv_CA(ch_in=64, ch_out=32)
        self.Up_conv1 = Conv_block_CA(ch_in=96, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d3 = self.Up3(x4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv_1x1(d1)
        
        return out


# 测试模型初始化
if __name__ == "__main__":
    # 模拟输入数据 [B, 1, 1600]
    dummy_input = torch.randn(8, 1, 1600)

    print("\n--- Testing CBAMUNet (Original First AUNet) ---")
    model_cbam = AUNet(ch_in=1, ch_out=1)
    output_cbam = model_cbam(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output_cbam.shape}")
    num_params_cbam = sum(p.numel() for p in model_cbam.parameters() if p.requires_grad)
    print(f"CBAMUNet 参数总数: {num_params_cbam:,}")