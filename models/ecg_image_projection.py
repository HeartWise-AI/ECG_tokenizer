"""Learned ECG-to-image projection for MedGemma integration.

This module converts raw ECG sequences (batch, num_leads, sequence_length)
into 3-channel square image tensors suitable for models that expect
`pixel_values` inputs (e.g., MedGemma). The transformation is trainable and
keeps gradients end-to-end so the ECG encoder and projection can be optimized
jointly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConvBlock(nn.Module):
    """A lightweight residual block for 2D feature refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 7),
        padding: Tuple[int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


@dataclass
class ECGImageProjectionConfig:
    """Configuration container for ECG2ImageProjection."""

    target_size: int = 896
    out_channels: int = 3
    base_channels: int = 32
    hidden_channels: Iterable[int] = (32, 64, 128, 256)
    dropout: float = 0.0
    output_activation: str = "sigmoid"  # "sigmoid", "tanh", or "none"
    expected_channels: int | None = None
    expected_length: int | None = None
    input_layout: str = "channels_first"  # or "channels_last"


class ECG2ImageProjection(nn.Module):
    """Project ECG sequences into square images for multimodal LLMs."""

    def __init__(
        self,
        config: ECGImageProjectionConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or ECGImageProjectionConfig()
        self._expected_channels = self.config.expected_channels
        self._expected_length = self.config.expected_length

        channels = [self.config.base_channels, *self.config.hidden_channels]

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(_ResidualConvBlock(in_ch, out_ch, dropout=self.config.dropout))
        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Conv2d(channels[-1], self.config.out_channels, kernel_size=1)

        activation = self.config.output_activation.lower()
        if activation == "sigmoid":
            self._activation = torch.sigmoid
        elif activation == "tanh":
            self._activation = torch.tanh
        elif activation == "none":
            self._activation = lambda x: x
        else:
            raise ValueError(
                "output_activation must be one of {'sigmoid', 'tanh', 'none'}"
            )

    def forward(self, ecg_signal: torch.Tensor) -> torch.Tensor:
        """Return differentiable image representation of the ECG signal."""

        if ecg_signal.dim() != 3:
            raise ValueError(
                "Expected tensor of shape (batch, channels, sequence_length) or (batch, sequence_length, channels)"
            )
        if self.config.input_layout not in {"channels_first", "channels_last"}:
            raise ValueError("input_layout must be 'channels_first' or 'channels_last'")

        if self.config.input_layout == "channels_last":
            ecg_signal = ecg_signal.transpose(1, 2)

        channels = ecg_signal.size(1)
        length = ecg_signal.size(2)

        if self._expected_channels is None:
            self._expected_channels = channels
        elif channels != self._expected_channels:
            raise ValueError(
                f"Projection expected {self._expected_channels} feature channels, got {channels}."
            )

        if self._expected_length is None:
            self._expected_length = length
        elif length != self._expected_length:
            raise ValueError(
                f"Projection expected sequence length {self._expected_length}, got {length}."
            )

        x = ecg_signal.unsqueeze(1)  # (B, 1, channels, seq)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = F.interpolate(
            x,
            size=(self.config.target_size, self.config.target_size),
            mode="bicubic",
            align_corners=False,
        )

        x = self.out_proj(x)
        x = self._activation(x)
        return x


__all__ = ["ECG2ImageProjection", "ECGImageProjectionConfig"]
