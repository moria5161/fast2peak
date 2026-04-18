import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models import register_model


def position_embedding(embed_dim, num_patches):
    pos = np.arange(num_patches, dtype=np.float64).reshape(-1, 1)
    dim_t = np.arange(embed_dim // 2, dtype=np.float64)
    dim_t = 1. / (10000.**(dim_t / (embed_dim // 2)))
    pos_times_dim = pos * dim_t
    emb = np.concatenate((np.sin(pos_times_dim), np.cos(pos_times_dim)), axis=1)
    if embed_dim % 2 != 0:
        emb = np.concatenate((emb, np.zeros((num_patches, 1))), axis=1)
    return torch.from_numpy(emb).float()


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)  


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads)**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape)
        return self.to_out(out)


class RelativeAttention(nn.Module):
    def __init__(self, dim, num_patches, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads)**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * num_patches - 1, heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords = torch.arange(num_patches)
        relative_coords = coords[None, :] - coords[:, None]
        relative_coords += num_patches - 1
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        dots = dots + relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RPEBlock(nn.Module):
    def __init__(self, dim, num_patches, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativeAttention(dim, num_patches, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=ks, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, channels, ks=3, dropout=0.2):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.enc_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i+1], ks=ks, dropout=dropout)
            for i in range(len(channels)-1)
        ])

    def forward(self, x):
        skips = []
        for block in self.enc_blocks:
            skips.append(x)
            x = self.pool(x)
            x = block(x)
        return skips, x


class Decoder(nn.Module):
    def __init__(self, channels, ks=3, dropout=0.2):
        super().__init__()
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose1d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels)-1)
        ])
        self.dec_blocks = nn.ModuleList([
            ConvBlock(channels[i+1] * 2, channels[i+1], ks=ks, dropout=dropout)
            for i in range(len(channels)-1)
        ])

    def forward(self, x, skips):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[i]
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)
        return x


@register_model
class Fast2PeakConv(nn.Module):
    def __init__(self, ks=3, dropout=0.2, **kwargs):
        super().__init__()
        channels = [32, 64, 128, 256]
        
        self.startconv = ConvBlock(1, channels[0], ks=ks, dropout=0.0)
        self.encoder = Encoder(channels, ks=ks, dropout=dropout)
        self.decoder = Decoder(channels[::-1], ks=ks, dropout=dropout)
        self.head = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x):
        x = self.startconv(x)
        skips, x = self.encoder(x)
        x = self.decoder(x, skips[::-1])
        return self.head(x)


@register_model
class Fast2Peak(nn.Module):
    def __init__(self, spectrum_length, patch_size, embed_dim, depth, heads, **kwargs):
        super().__init__()
        if spectrum_length % patch_size != 0:
            raise ValueError("The spectral length must be divisible by the size of the patch")
        self.num_patches = spectrum_length // patch_size
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(position_embedding(embed_dim, self.num_patches), requires_grad=False)
        self.transformer_blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.decoder = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 2),
                            nn.GELU(),
                            nn.Linear(embed_dim * 2, patch_size)
                                    )
        self.patch_size = patch_size

    def unpatchify(self, patches):
        B, n_patches, p = patches.shape
        C = 1
        L = n_patches * p
        return patches.reshape(B, n_patches, C, p).permute(0, 2, 1, 3).reshape(B, C, L)

    def forward(self, x):
        patches = self.patch_embed(x) + self.pos_embed
        for blk in self.transformer_blocks:
            patches = blk(patches)
        pred_patches = self.decoder(patches)
        return self.unpatchify(pred_patches)


@register_model
class Fast2PeakRPE(nn.Module):
    def __init__(self, spectrum_length, patch_size, embed_dim, depth, heads, **kwargs):
        super().__init__()
        if spectrum_length % patch_size != 0:
            raise ValueError("The spectral length must be divisible by the size of the patch")
        
        self.num_patches = spectrum_length // patch_size
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            RPEBlock(embed_dim, self.num_patches, heads) for _ in range(depth)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size)
        )
        self.patch_size = patch_size

    def unpatchify(self, patches):
        B, n_patches, p = patches.shape
        C = 1
        L = n_patches * p
        return patches.reshape(B, n_patches, C, p).permute(0, 2, 1, 3).reshape(B, C, L)

    def forward(self, x):
        patches = self.patch_embed(x)
        
        for blk in self.transformer_blocks:
            patches = blk(patches)
            
        pred_patches = self.decoder(patches)
        return self.unpatchify(pred_patches)


@register_model
class Fast2PeakOverlap(nn.Module):
    def __init__(self, spectrum_length, patch_size, embed_dim, depth, heads, overlap_ratio=0.25, **kwargs):
        super().__init__()
        self.spectrum_length = spectrum_length
        self.patch_size = patch_size
        
        self.stride = int(patch_size * (1 - overlap_ratio))
        
        if (spectrum_length - patch_size) % self.stride != 0:
            raise ValueError(f"Error: {spectrum_length} and {patch_size} with stride {self.stride}")
            
        self.num_patches = (spectrum_length - patch_size) // self.stride + 1
        
        self.patch_embed = OverlapPatchEmbed(patch_size, self.stride, embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            RPEBlock(embed_dim, self.num_patches, heads) for _ in range(depth)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size)
        )

        self.register_buffer('divisor', self._compute_divisor())

    def _compute_divisor(self):
        ones = torch.ones(1, self.patch_size, self.num_patches)
        divisor_2d = F.fold(
            ones, 
            output_size=(1, self.spectrum_length), 
            kernel_size=(1, self.patch_size), 
            stride=(1, self.stride)
        )
        return divisor_2d.squeeze(2)

    def unpatchify(self, patches):
        patches = patches.transpose(1, 2)
        
        output_2d = F.fold(
            patches,
            output_size=(1, self.spectrum_length),
            kernel_size=(1, self.patch_size),
            stride=(1, self.stride)
        )
        output_1d = output_2d.squeeze(2)
        
        return output_1d / self.divisor

    def forward(self, x):
        patches = self.patch_embed(x)
        for blk in self.transformer_blocks:
            patches = blk(patches)
        pred_patches = self.decoder(patches)
        return self.unpatchify(pred_patches)


if __name__ == "__main__":
    from thop import profile
    # net = Fast2PeakConv()
    # net = Fast2Peak(spectrum_length=1600, patch_size=16, embed_dim=256, depth=8, heads=8)
    net = Fast2PeakRPE(spectrum_length=1600, patch_size=16, embed_dim=256, depth=8, heads=8)
    # net = Fast2PeakOverlap(spectrum_length=1600, patch_size=16, embed_dim=256, depth=8, heads=8)
    x = torch.randn(1, 1, 1600)
    y = net(x)
    print(y.shape)

    macs, params = profile(net, inputs=(x,))

    print(f'params: {params/1e6} M, macs: {macs/1e9} G') 