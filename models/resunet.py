import torch
from torch import nn
import torch.nn.functional as F

from . import register_model


class BasicConv(nn.Module):
    def __init__(self, channels_in, channels_out, batch_norm):
        super(BasicConv, self).__init__()
        modules = [nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True)]
        modules.append(nn.PReLU())
        if batch_norm:
            modules.append(nn.BatchNorm1d(channels_out))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)

class ResUNetConv(nn.Module):
    def __init__(self, num_convs, channels, batch_norm):
        super(ResUNetConv, self).__init__()
        unet_conv = []
        for _ in range(num_convs):
            unet_conv.append(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
            unet_conv.append(nn.PReLU())
            if batch_norm:
                unet_conv.append(nn.BatchNorm1d(channels))
        self.body = nn.Sequential(*unet_conv)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


@register_model
class ResUNet(nn.Module):
    def __init__(self, num_convs=6, batch_norm=True, **kwargs):
        super(ResUNet, self).__init__()
        
        # Encoder Stage 1 (1600 -> 800)
        self.conv1 = nn.Sequential(
            BasicConv(1, 64, batch_norm),
            ResUNetConv(num_convs, 64, batch_norm)
        )
        self.pool1 = nn.MaxPool1d(2)

        # Encoder Stage 2 (800 -> 400)
        self.conv2 = nn.Sequential(
            BasicConv(64, 128, batch_norm),
            ResUNetConv(num_convs, 128, batch_norm)
        )
        self.pool2 = nn.MaxPool1d(2)

        # Bridge (400)
        self.conv3 = nn.Sequential(
            BasicConv(128, 256, batch_norm),
            ResUNetConv(num_convs, 256, batch_norm),
            BasicConv(256, 128, batch_norm)
        )
        self.up3 = nn.Upsample(scale_factor=2) 

        # Decoder Stage 1 (800 -> 1600)
        self.conv4 = nn.Sequential(
            BasicConv(256, 128, batch_norm),
            ResUNetConv(num_convs, 128, batch_norm),
            BasicConv(128, 64, batch_norm)
        )
        self.up4 = nn.Upsample(scale_factor=2)

        # Decoder Stage 2
        self.conv5 = nn.Sequential(
            BasicConv(128, 64, batch_norm),
            ResUNetConv(num_convs, 64, batch_norm)
        )
        
        self.conv6 = BasicConv(64, 1, batch_norm)
        
        # Final refinement using 1x1 convolutions
        self.final = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )
                
    def forward(self, x):
        # Encoder
        x1_out = self.conv1(x)       # [B, 64, 1600]
        x1_pool = self.pool1(x1_out) # [B, 64, 800]
        
        x2_out = self.conv2(x1_pool) # [B, 128, 800]
        x2_pool = self.pool2(x2_out) # [B, 128, 400]
        
        # Bridge
        x3 = self.conv3(x2_pool)     # [B, 128, 400]
        x3_up = self.up3(x3)         # [B, 128, 800]
        
        # Decoder
        x4 = torch.cat((x2_out, x3_up), dim=1) # [B, 256, 800]
        x4 = self.conv4(x4)                    # [B, 64, 800]
        x5_up = self.up4(x4)                   # [B, 64, 1600]

        # Standard U-Net Skip Connection
        x6 = torch.cat((x1_out, x5_up), dim=1) # [B, 128, 1600]
        x6 = self.conv5(x6)                    # [B, 64, 1600]
        x7 = self.conv6(x6)                    # [B, 1, 1600]
        
        out = self.final(x7)
        return out

if __name__ == "__main__":
    model = ResUNet(num_convs=6, batch_norm=True)
    dummy_input = torch.randn(8, 1, 1600)
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")