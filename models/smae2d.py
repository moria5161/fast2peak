import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ViT_1D import Block, get_1d_sincos_pos_embed_from_grid
from models import register_model


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """2D sin-cos positional embedding for a square grid of patches."""
    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2

    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid_size)
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid_size)

    pos = np.arange(grid_size * grid_size)
    row = pos // grid_size
    col = pos % grid_size

    emb = np.concatenate([emb_h[row], emb_w[col]], axis=1)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)

    return emb


class MaskedAutoencoder2DViT(nn.Module):
    """Masked Autoencoder with 2D patch extraction + ViT backbone (SMAE baseline)."""

    def __init__(self, spectra_size=1600, patch_size=16, encoder_dim=256, depth=8, num_heads=8,
                 decoder_embed_dim=256, decoder_depth=1, decoder_num_heads=8, dim_mlp=256,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.image_size = int(spectra_size ** 0.5)
        self.patch_side = int(patch_size ** 0.5)
        assert self.image_size ** 2 == spectra_size, \
            "spectra_size must be a perfect square (e.g. 1600)"
        assert self.patch_side ** 2 == patch_size, \
            "patch_size must be a perfect square (e.g. 16)"

        self.patch_size = patch_size
        self.num_patches = (self.image_size // self.patch_side) ** 2

        # --- Encoder ---
        self.patch_embed = nn.Linear(patch_size, encoder_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, encoder_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList([
            Block(dim_in=encoder_dim, heads=num_heads, dim_head=int(encoder_dim // num_heads),
                  dim_mlp=dim_mlp)
            for _ in range(depth)
        ])
        self.norm = norm_layer(encoder_dim)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(dim_in=decoder_embed_dim, heads=decoder_num_heads,
                  dim_head=int(decoder_embed_dim // decoder_num_heads), dim_mlp=dim_mlp)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.image_size // self.patch_side,
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.image_size // self.patch_side,
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, spectra):
        """Extract 2D patches from 1D spectra. [B, 1, L] -> [B, num_patches, patch_size]"""
        B = spectra.shape[0]
        img = spectra.reshape(B, 1, self.image_size, self.image_size)
        patches = F.unfold(img, kernel_size=self.patch_side, stride=self.patch_side)
        patches = patches.transpose(1, 2)
        return patches

    def unpatchify(self, x):
        """Reconstruct 1D spectra from 2D patches. [B, num_patches, patch_size] -> [B, 1, L]"""
        B = x.shape[0]
        patches = x.transpose(1, 2)
        img = F.fold(patches, output_size=(self.image_size, self.image_size),
                     kernel_size=self.patch_side, stride=self.patch_side)
        spectra = img.reshape(B, 1, -1)
        return spectra

    def random_masking(self, x, mask_ratio):
        N = x.shape[0]
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patchify(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, spectra, pred, mask):
        target = self.patchify(spectra)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, spectra, mask_ratio=0.50):
        latent, mask, ids_restore = self.forward_encoder(spectra, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(spectra, pred, mask)
        return loss, pred, mask


@register_model
def S2DMAE(**kwargs):
    return spectraMAE2D_spectra_size1600_patch16_dim256_h8(**kwargs)


def spectraMAE2D_spectra_size1600_patch16_dim256_h8(**kwargs):
    return MaskedAutoencoder2DViT(
        spectra_size=1600, patch_size=16, encoder_dim=256, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=1, decoder_num_heads=8, dim_mlp=256,
    )
