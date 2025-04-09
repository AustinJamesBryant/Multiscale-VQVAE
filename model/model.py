# Model from:
#   flexvar: https://github.com/jiaosiyu1999/FlexVAR
# Please view to see additional acknowledements
from dataclasses import dataclass, field
from typing import List, Tuple


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lpips
from .modules import Encoder, VectorQuantizer, Decoder


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


class MultiscaleVQVAE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.codebook_size  # For convenience if needed
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.quantize = VectorQuantizer(
            config.codebook_size, config.codebook_embed_dim,
            config.commit_loss_beta, config.entropy_loss_ratio,
            config.codebook_l2_norm, config.codebook_show_usage
        )
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        print(f"VQ Model Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def encode_conti(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def multi_scale_encode(self, x, v_patch_nums):
        h = self.encoder(x)
        all_quant, all_emb_loss = [], []
        for hw in v_patch_nums:
            _h = F.interpolate(h.clone(), size=(hw, hw), mode='area')
            _h = self.quant_conv(_h)
            quant, emb_loss, _ = self.quantize(_h)
            all_quant.append(quant)
            all_emb_loss.append(emb_loss)
        return all_quant, all_emb_loss

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, v_patch_nums=None):
        if v_patch_nums is None:
            quant, diff, _ = self.encode(input)
            dec = self.decode(quant)
        else:
            all_quant, diff = self.multi_scale_encode(input, v_patch_nums)
            dec = [self.decode(q) for q in all_quant]
        return dec, diff

class LitMultiscaleVQVAE(pl.LightningModule):
    def __init__(
        self,
        model_args: ModelArgs,
        patch_nums: Tuple[int, ...] = (1, 2, 4, 8, 16),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_args'])
        self.vqvae = MultiscaleVQVAE(model_args)
        self.learning_rate = learning_rate

        # For training we use a dynamic patch scale, for validation a fixed one.
        self.fixed_patch_nums = patch_nums

    def gen_curr_patch_nums(self):
        max_patch = self.fixed_patch_nums[-1]
        
        # Always include 1, 2, and max_patch.
        # Sample 7 additional unique numbers from 3 to max_patch-1.
        if max_patch < 3:
            raise ValueError("max_patch must be at least 3 to allow for additional levels.")
        
        mid_candidates = list(range(3, max_patch))
        if len(mid_candidates) < 7:
            raise ValueError("Not enough values between 3 and max_patch-1 to sample 7 unique levels.")
        
        extra_patches = random.sample(mid_candidates, 7)
        curr_patch_nums = sorted([1, 2] + extra_patches + [max_patch])
        
        # Calculate the threshold as 70% of the total fixed patch sums (sum of squares)
        fixed_threshold = sum(pn ** 2 for pn in self.fixed_patch_nums) * 0.7
        total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        
        # Randomly drop one element at a time from the middle
        # (always keeping the first two elements and the last element)
        while total_lens > fixed_threshold and len(curr_patch_nums) > 3:
            # Define candidate indices: skip the first two and the last element.
            candidate_indices = list(range(2, len(curr_patch_nums) - 1))
            if not candidate_indices:
                break  # No middle element to remove.
            drop_index = random.choice(candidate_indices)
            curr_patch_nums.pop(drop_index)
            total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        
        return curr_patch_nums

    def training_step(self, batch, batch_idx):
        x, _ = batch
        curr_patch_nums = self.gen_curr_patch_nums()
        recons, diff_list = self.vqvae(x, v_patch_nums=curr_patch_nums)
        
        # Use the input resolution as target size.
        target_size = x.shape[-2:]
        recon_loss = 0.0
        for recon in recons:
            recon_up = F.interpolate(recon, size=target_size, mode='bicubic')
            recon_loss += F.mse_loss(recon_up, x)
        recon_loss = recon_loss / len(recons)
        
        processed_diff = []
        for d in diff_list:
            if d is None:
                processed_diff.append(torch.tensor(0.0, device=x.device))
            elif isinstance(d, tuple):
                val = d[0] if d[0] is not None else torch.tensor(0.0, device=x.device)
                processed_diff.append(val)
            else:
                processed_diff.append(d)
        quant_loss = sum(processed_diff) / len(processed_diff) if processed_diff else torch.tensor(0.0, device=x.device)
        
        loss = recon_loss + quant_loss

        # Log losses
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_quant_loss", quant_loss, prog_bar=True)
        
        # Log current learning rate to the progress bar
        # Access the learning rate from the optimizer's parameter group.
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recons, diff_list = self.vqvae(x, v_patch_nums=self.fixed_patch_nums)
        target_size = x.shape[-2:]
        recon_loss = 0.0
        for recon in recons:
            recon_up = F.interpolate(recon, size=target_size, mode='bicubic')
            recon_loss += F.mse_loss(recon_up, x)
        recon_loss = recon_loss / len(recons)
        
        processed_diff = []
        for d in diff_list:
            if d is None:
                processed_diff.append(torch.tensor(0.0, device=x.device))
            elif isinstance(d, tuple):
                val = d[0] if d[0] is not None else torch.tensor(0.0, device=x.device)
                processed_diff.append(val)
            else:
                processed_diff.append(d)
        quant_loss = sum(processed_diff) / len(processed_diff) if processed_diff else torch.tensor(0.0, device=x.device)
        
        loss = recon_loss + quant_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_quant_loss", quant_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class FinetuneLitMultiscaleVQVAE(pl.LightningModule): 
    def __init__(
        self,
        model_args: ModelArgs,
        patch_nums: Tuple[int, ...] = (1, 2, 4, 8, 16),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_args'])
        self.vqvae = MultiscaleVQVAE(model_args)
        self.learning_rate = learning_rate
        self.fixed_patch_nums = patch_nums

        # Initialize the LPIPS module and freeze its weights
        self.lpips_fn = lpips.LPIPS(net='vgg')
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def gen_curr_patch_nums(self):
        max_patch = self.fixed_patch_nums[-1]
        
        # Always include 1, 2, and max_patch.
        # Sample 7 additional unique numbers from 3 to max_patch-1.
        if max_patch < 3:
            raise ValueError("max_patch must be at least 3 to allow for additional levels.")
        
        mid_candidates = list(range(3, max_patch))
        if len(mid_candidates) < 7:
            raise ValueError("Not enough values between 3 and max_patch-1 to sample 7 unique levels.")
        
        extra_patches = random.sample(mid_candidates, 7)
        curr_patch_nums = sorted([1, 2] + extra_patches + [max_patch])
        
        # Calculate the threshold as 70% of the total fixed patch sums (sum of squares)
        fixed_threshold = sum(pn ** 2 for pn in self.fixed_patch_nums) * 0.7
        total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        
        # Randomly drop one element at a time from the middle
        # (always keeping the first two elements and the last element)
        while total_lens > fixed_threshold and len(curr_patch_nums) > 3:
            # Define candidate indices: skip the first two and the last element.
            candidate_indices = list(range(2, len(curr_patch_nums) - 1))
            if not candidate_indices:
                break  # No middle element to remove.
            drop_index = random.choice(candidate_indices)
            curr_patch_nums.pop(drop_index)
            total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        
        return curr_patch_nums

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Convert the input from [0, 1] to [-1, 1]
        x_norm = x * 2.0 - 1.0

        curr_patch_nums = self.gen_curr_patch_nums()
        recons, diff_list = self.vqvae(x, v_patch_nums=curr_patch_nums)
        target_size = x_norm.shape[-2:]
        
        recon_loss = 0.0
        recon_mse_loss = 0.0
        for recon in recons:
            # Resize the reconstruction and convert to [-1, 1]
            recon_up = F.interpolate(recon, size=target_size, mode='bicubic')
            recon_up_norm = recon_up * 2.0 - 1.0
            lpips_loss = self.lpips_fn(recon_up_norm, x_norm)
            recon_mse_loss += F.mse_loss(recon_up, x)
            recon_loss += lpips_loss.mean()
        recon_loss /= len(recons)
        recon_mse_loss /= len(recons)

        # Process quantization-related differences
        processed_diff = []
        for d in diff_list:
            if d is None:
                processed_diff.append(torch.tensor(0.0, device=x.device))
            elif isinstance(d, tuple):
                val = d[0] if d[0] is not None else torch.tensor(0.0, device=x.device)
                processed_diff.append(val)
            else:
                processed_diff.append(d)
        quant_loss = (
            sum(processed_diff) / len(processed_diff)
            if processed_diff
            else torch.tensor(0.0, device=x.device)
        )

        loss = recon_loss + recon_mse_loss + quant_loss

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_mse_loss", recon_mse_loss, prog_bar=True)
        self.log("train_recon_lpips_loss", recon_loss, prog_bar=True)
        self.log("train_quant_loss", quant_loss, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Convert the input from [0, 1] to [-1, 1]
        x_norm = x * 2.0 - 1.0

        recons, diff_list = self.vqvae(x, v_patch_nums=self.fixed_patch_nums)
        target_size = x_norm.shape[-2:]
        
        recon_loss = 0.0
        recon_mse_loss = 0.0
        for recon in recons:
            # Resize the reconstruction and convert to [-1, 1]
            recon_up = F.interpolate(recon, size=target_size, mode='bicubic')
            recon_up_norm = recon_up * 2.0 - 1.0
            lpips_loss = self.lpips_fn(recon_up_norm, x_norm)
            recon_mse_loss += F.mse_loss(recon_up, x)
            recon_loss += lpips_loss.mean()
        recon_loss /= len(recons)
        recon_mse_loss /= len(recons)

        # Process quantization-related differences
        processed_diff = []
        for d in diff_list:
            if d is None:
                processed_diff.append(torch.tensor(0.0, device=x.device))
            elif isinstance(d, tuple):
                val = d[0] if d[0] is not None else torch.tensor(0.0, device=x.device)
                processed_diff.append(val)
            else:
                processed_diff.append(d)
        quant_loss = (
            sum(processed_diff) / len(processed_diff)
            if processed_diff
            else torch.tensor(0.0, device=x.device)
        )

        loss = recon_loss + recon_mse_loss + quant_loss

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_mse_loss", recon_mse_loss, prog_bar=True)
        self.log("val_recon_lpips_loss", recon_loss, prog_bar=True)
        self.log("val_quant_loss", quant_loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer