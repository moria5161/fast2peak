#!/usr/bin/env python3
"""PyTorch equivalent of the authors' TensorLayer DN-Unet inference model.

The official model uses 2-D convolutions on tensors shaped [B, 1, 8192, 1].
Because the height is always one, TensorFlow SAME padding selects kernel row 1;
the implementation below is the exactly corresponding 1-D network.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, batch_norm: bool, preactivate: bool):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm1d(channels_out, eps=1e-5) if batch_norm else None
        self.preactivate = preactivate

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv(values)
        if self.preactivate:
            values = F.leaky_relu(values, negative_slope=0.2)
        if self.batch_norm is not None:
            values = self.batch_norm(values)
            values = F.leaky_relu(values, negative_slope=0.2)
        return values


class DecoderLayer(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            channels_in, channels_out, kernel_size=4, stride=2, padding=1
        )
        self.batch_norm = nn.BatchNorm1d(channels_out, eps=1e-5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.batch_norm(self.deconv(values)), negative_slope=0.2)


class DNUNet(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_in = [1, 64, 128, 256] + [512] * 8
        encoder_out = [64, 128, 256] + [512] * 9
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    channels_in,
                    channels_out,
                    batch_norm=1 <= index <= 10,
                    preactivate=7 <= index <= 11,
                )
                for index, (channels_in, channels_out) in enumerate(
                    zip(encoder_in, encoder_out)
                )
            ]
        )
        decoder_pairs = [
            (512, 512),
            (1024, 512),
            (1024, 512),
            (1024, 512),
            (1024, 512),
            (1024, 1024),
            (1536, 1024),
            (1536, 1024),
            (1536, 256),
            (512, 128),
            (256, 64),
            (128, 64),
        ]
        self.decoder = nn.ModuleList(
            [DecoderLayer(channels_in, channels_out) for channels_in, channels_out in decoder_pairs]
        )
        self.output = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skips = []
        for layer in self.encoder:
            values = layer(values)
            skips.append(values)
        for index, layer in enumerate(self.decoder):
            if index:
                values = torch.cat([values, skips[-1 - index]], dim=1)
            values = layer(values)
        return F.leaky_relu(self.output(values), negative_slope=0.2)


def _conv1d_weight(tf_weight: np.ndarray, transpose: bool = False) -> torch.Tensor:
    if tf_weight.ndim != 4 or tf_weight.shape[0:2] not in ((4, 4), (1, 1)):
        raise ValueError(f"unexpected TensorFlow convolution shape: {tf_weight.shape}")
    height_index = 1 if tf_weight.shape[0] == 4 else 0
    effective = tf_weight[height_index]
    if transpose:
        # TensorFlow deconv: [width, out_channels, in_channels]
        effective = effective.transpose(2, 1, 0)
    else:
        # TensorFlow conv: [width, in_channels, out_channels]
        effective = effective.transpose(2, 1, 0)
    return torch.from_numpy(np.ascontiguousarray(effective)).float()


def _assign_conv(layer: nn.Conv1d, params: np.ndarray, index: int) -> int:
    weight, bias = params[index], params[index + 1]
    converted = _conv1d_weight(weight, transpose=False)
    if converted.shape != layer.weight.shape or bias.shape != tuple(layer.bias.shape):
        raise ValueError(
            f"DN-Unet conv mismatch at {index}: {converted.shape}/{bias.shape} versus "
            f"{tuple(layer.weight.shape)}/{tuple(layer.bias.shape)}"
        )
    layer.weight.data.copy_(converted)
    layer.bias.data.copy_(torch.from_numpy(np.asarray(bias)).float())
    return index + 2


def _assign_deconv(layer: nn.ConvTranspose1d, params: np.ndarray, index: int) -> int:
    weight, bias = params[index], params[index + 1]
    converted = _conv1d_weight(weight, transpose=True)
    if converted.shape != layer.weight.shape or bias.shape != tuple(layer.bias.shape):
        raise ValueError(
            f"DN-Unet deconv mismatch at {index}: {converted.shape}/{bias.shape} versus "
            f"{tuple(layer.weight.shape)}/{tuple(layer.bias.shape)}"
        )
    layer.weight.data.copy_(converted)
    layer.bias.data.copy_(torch.from_numpy(np.asarray(bias)).float())
    return index + 2


def _assign_batch_norm(layer: nn.BatchNorm1d, params: np.ndarray, index: int) -> int:
    # TensorLayer 1.11 stores beta, gamma, moving_mean, moving_variance.
    beta, gamma, moving_mean, moving_variance = params[index : index + 4]
    expected = tuple(layer.weight.shape)
    if any(array.shape != expected for array in (beta, gamma, moving_mean, moving_variance)):
        raise ValueError(f"DN-Unet batch-norm mismatch at {index}")
    layer.bias.data.copy_(torch.from_numpy(np.asarray(beta)).float())
    layer.weight.data.copy_(torch.from_numpy(np.asarray(gamma)).float())
    layer.running_mean.data.copy_(torch.from_numpy(np.asarray(moving_mean)).float())
    layer.running_var.data.copy_(torch.from_numpy(np.asarray(moving_variance)).float())
    return index + 4


def convert_official_npz(source_npz: Path, destination: Path) -> dict:
    print(f"Converting official DN-Unet TensorLayer weights: {source_npz}", flush=True)
    packed = np.load(source_npz, allow_pickle=True)
    params = packed["params"]
    if params.shape != (138,):
        raise ValueError(f"expected 138 official DN-Unet arrays, got {params.shape}")
    model = DNUNet()
    index = 0
    for layer in model.encoder:
        index = _assign_conv(layer.conv, params, index)
        if layer.batch_norm is not None:
            index = _assign_batch_norm(layer.batch_norm, params, index)
    for layer in model.decoder:
        index = _assign_deconv(layer.deconv, params, index)
        index = _assign_batch_norm(layer.batch_norm, params, index)
    index = _assign_conv(model.output, params, index)
    if index != len(params):
        raise ValueError(f"consumed {index} DN-Unet arrays, expected {len(params)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "source_npz": str(source_npz),
        "source_array_count": int(len(params)),
        "tensorflow_same_effective_kernel_row": 1,
        "native_length": 8192,
    }
    torch.save(payload, destination)
    del params, packed, model
    gc.collect()
    print(f"Saved converted DN-Unet checkpoint: {destination}", flush=True)
    return payload


def load_dn_unet(
    source_npz: Path,
    converted_checkpoint: Path,
    device: torch.device,
    trained_checkpoint: Path | None = None,
) -> DNUNet:
    if trained_checkpoint is None and not converted_checkpoint.is_file():
        convert_official_npz(source_npz, converted_checkpoint)
    checkpoint_path = trained_checkpoint or converted_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DNUNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model
