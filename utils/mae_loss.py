import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, weight_mae=1.0):
        super().__init__()
        self.weight_mae = weight_mae
        self.mae_loss = nn.L1Loss(reduction='mean')
        
    def forward(self, loss_patch, loss_output):
        """
        loss_patch: [B, num_masks, patch_area] (True masked target patches)
        loss_output: [B, num_masks, patch_area] (Predicted masked patches)
        """
        # Calculate L1 (MAE) loss
        mae = self.mae_loss(loss_output, loss_patch)
        
        # Calculate cosine loss (axis=1 in tf translates across the patch dimension)
        # Flatten the masks for each batch
        B = loss_patch.shape[0]
        t_flat = loss_patch.view(B, -1)
        p_flat = loss_output.view(B, -1)
        
        # tf.keras.losses.CosineSimilarity reduction='sum' effectively sums the batch
        # We'll use sum here then average if required.
        cos_sim = F.cosine_similarity(t_flat, p_flat, dim=1)
        
        # -1 * cos_sim + 1e-7
        cosine_loss = -cos_sim.sum() + 1e-7
        # Note: Depending on your batch processing you might want mean().
        # Original notebook: cosine_loss = -cosine_similarity(y_true, y_pred) + 1E-7
        # We will use mean to stay invariant to batch size for stability
        cosine_loss = (-1 * cos_sim).mean() + 1e-7
        
        return (self.weight_mae * mae) + (((1 - self.weight_mae) * cosine_loss) + 1e-6)
