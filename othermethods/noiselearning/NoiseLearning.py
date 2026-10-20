from abc import ABC
import torch
import torch.nn as nn
import os
from scipy.fftpack import dct, idct
from scipy.signal import resample


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


class UNet_CA(nn.Module, ABC):
    def __init__(self, ch_in=1, ch_out=1):
        super(UNet_CA, self).__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block_CA(ch_in=ch_in, ch_out=64)
        self.Conv2 = Conv_block_CA(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block_CA(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block_CA(ch_in=256, ch_out=512)
        self.Conv5 = Conv_block_CA(ch_in=512, ch_out=1024)

        self.Up5 = Up_conv_CA(ch_in=1024, ch_out=512)
        self.Up_conv5 = Conv_block_CA(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv_CA(ch_in=512, ch_out=256)
        self.Up_conv4 = Conv_block_CA(ch_in=512, ch_out=256)

        self.Up3 = Up_conv_CA(ch_in=256, ch_out=128)
        self.Up_conv3 = Conv_block_CA(ch_in=256, ch_out=128)

        self.Up2 = Up_conv_CA(ch_in=128, ch_out=64)
        self.Up_conv2 = Conv_block_CA(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

def NL(noisy_spec, model_pretrained_path, scale_factor=100):
    orig_len = len(noisy_spec)
    # DCT 变换
    noisy_spec = resample(noisy_spec, 1600) / scale_factor
    noisy_spec_dct = dct(noisy_spec, norm='ortho')

    model = UNet_CA(1, 1)
    model_file = os.path.join(model_pretrained_path)
    # 加载模型参数
    state = torch.load(model_file)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 预测
    output = model(torch.from_numpy(noisy_spec_dct.reshape(1, 1, -1)).float().cuda())
    output = output.detach().cpu().numpy().squeeze()

    # 逆 DCT 变换
    output = idct(output, norm='ortho')
    denoised_spec = noisy_spec - output
    denoised_spec = resample(denoised_spec, orig_len)
    return denoised_spec * scale_factor



if __name__ == "__main__":
    # 测试代码
    import numpy as np
    import time
    import os
    import argparse
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', default=str(project_root / 'experiments/noiselearning'))
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parent / 'checkpoints/562800.pt'))
    args = parser.parse_args()
    test_dir = args.input_dir
    model_pretrained_path = args.checkpoint
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(test_dir):
        if filename.endswith('.txt'):
            noisy_spec = np.loadtxt(os.path.join(test_dir, filename))[:, 1]
            waves = np.loadtxt(os.path.join(test_dir, filename))[:, 0]
            print(f'Processing {filename}...')

            for scale_factor in range(50, 100, 10):  # 50 到 100，每隔 10 一次
                start_time = time.time()
                denoised_spec = NL(noisy_spec, model_pretrained_path, scale_factor=scale_factor)
                outname = filename.replace('.txt', f'_NL_{scale_factor}.txt')
                np.savetxt(os.path.join(output_dir, outname), np.column_stack((waves, denoised_spec)), fmt='%.6f')
                print(f'  scale={scale_factor} done in {time.time() - start_time:.2f}s')

    print('All files processed successfully.')
