import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_model
from utils.mae_loss import CombinedLoss

class Patches2D(nn.Module):
    def __init__(self, image_size=20, patch_side=5):
        super().__init__()
        self.image_size = image_size
        self.patch_side = patch_side
        self.num_patches = (image_size // patch_side) ** 2
        self.patch_area = patch_side * patch_side

    def vec_to_image(self, vec):
        B = vec.shape[0]
        # In the original code, each chunk becomes a patch directly.
        # Patch ordering is row-major.
        patches = vec.view(B, self.num_patches, self.patch_side, self.patch_side)
        
        # Reshape into an image
        patches = patches.view(B, 
                               self.image_size // self.patch_side, 
                               self.image_size // self.patch_side, 
                               self.patch_side, 
                               self.patch_side)
        
        # Permute to [B, H_patches, patch_H, W_patches, patch_W] -> [B, image_size, image_size]
        img = patches.permute(0, 1, 3, 2, 4).contiguous()
        img = img.view(B, 1, self.image_size, self.image_size)
        return img

    def extract_patches(self, images):
        B = images.shape[0]
        # Extract non-overlapping patches
        patches = F.unfold(images, kernel_size=self.patch_side, stride=self.patch_side)
        # Transpose to [B, num_patches, patch_area]
        patches = patches.transpose(1, 2)
        return patches

    def forward(self, vec):
        img = self.vec_to_image(vec)
        patches = self.extract_patches(img)
        return patches, img

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

@register_model
class RamanMAE(nn.Module):
    def __init__(self, spectrum_length=1600, patch_size=16, mask_ratio=0.5,
                 embed_dim=64, depth=3, heads=4,
                 dec_dim=32, dec_depth=1, dec_heads=4, **kwargs):
        super().__init__()
        
        # Original notebook logic mapping
        self.spectrum_length = spectrum_length
        self.image_size = int(math.sqrt(spectrum_length))
        self.patch_side = int(math.sqrt(patch_size))
        self.patch_area = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (self.image_size // self.patch_side) ** 2
        
        assert self.image_size ** 2 == spectrum_length, "spectrum_length must easily form a 2D square image."
        assert self.patch_side ** 2 == patch_size, "patch_size must be a perfect square."
        
        # 2D Patch Extraction
        self.patcher = Patches2D(self.image_size, self.patch_side)

        # Encoder
        self.enc_proj = nn.Linear(self.patch_area, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.enc_blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Decoder initialization
        self.dec_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.dec_pred = nn.Linear(self.num_patches * dec_dim, spectrum_length)
        
        # Internal Loss Function strictly from original author
        self.criterion = CombinedLoss(weight_mae=1.0)
    
    def unpatchify(self, vec):
        """
        Dummy logic. We natively produce a 1D vector at pre_final, so just return it.
        Or view it to [B, 1, spectrum_length] for the testing loop.
        """
        return vec.unsqueeze(1)
        
    def forward(self, vec, mask_ratio=None):
        """
        vec: [B, 1, spectrum_length] (from dataloader in the framework)
        """
        B = vec.shape[0]
        # Remove channel dimension for 1D to 2D magic
        vec_1d = vec.squeeze(1).float() 
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # 1. Convert to 2D image and extract patches
        patches, img = self.patcher(vec_1d) # patches: [B, num_patches, patch_area]
        
        # 2. Patch Encoding & Add Positional Embedding
        x = self.enc_proj(patches) + self.pos_embed
        
        # 3. Masking
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))
        if len_keep == L:
            # During test where mask_ratio=0.0
            ids_keep = torch.arange(L, device=vec.device).unsqueeze(0).expand(B, -1)
            ids_mask = torch.empty(B, 0, dtype=torch.long, device=vec.device)
            ids_restore = ids_keep
        else:
            noise = torch.rand(B, L, device=vec.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            ids_mask = ids_shuffle[:, len_keep:]
            
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        pos_kept = torch.gather(self.pos_embed.expand(B, -1, -1), dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        # 4. Encoder
        for blk in self.enc_blocks:
            x_kept = blk(x_kept)
        x_kept = self.enc_norm(x_kept)

        # 5. Append mask tokens
        if ids_mask.shape[1] > 0:
            pos_mask = torch.gather(self.pos_embed.expand(B, -1, -1), dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, x_kept.shape[2]))
            mask_tokens = self.mask_token.repeat(B, ids_mask.shape[1], 1)
            mask_tokens = mask_tokens + pos_mask
            x_kept = x_kept + pos_kept
            x_full = torch.cat([x_kept, mask_tokens], dim=1)
        else:
            x_kept = x_kept + pos_kept
            x_full = x_kept

        # Unshuffle to original positions
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]))

        # 6. Decoder embed
        x_full = self.dec_embed(x_full)

        # 7. Decoder
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        x_full = self.dec_norm(x_full)
        
        # Decoder predicts the WHOLE image (flattened 1D length)
        x_full = x_full.flatten(1) 
        pre_final = torch.sigmoid(self.dec_pred(x_full)) # [B, spectrum_length]
        
        # 8. Calculate Loss internally (only for training) if masks exist
        loss = torch.tensor(0.0, device=vec.device)
        if ids_mask.shape[1] > 0:
            decoder_output_img = pre_final.view(B, 1, self.image_size, self.image_size)
            decoder_patches = self.patcher.extract_patches(decoder_output_img)
            
            loss_patch = torch.gather(patches, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, patches.shape[2]))
            loss_output = torch.gather(decoder_patches, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, decoder_patches.shape[2]))
            
            loss = self.criterion(loss_patch, loss_output)
            
        # Returning shape similar to SMAE expectations `loss, output, _`
        return loss, pre_final, None
