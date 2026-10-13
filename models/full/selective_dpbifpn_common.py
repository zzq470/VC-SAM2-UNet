"""Shared selective dual-path BiFPN blocks for VC-DPBiFPN Stage 3."""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


NODE_SIZES = OrderedDict((('t22', 2), ('t44', 2), ('t88', 2), ('b44', 3), ('b22', 3)))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class AlignProjection(nn.Module):
    def __init__(self, in_channels: int, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class SeparableNode(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.depthwise(x)))
        return self.act(self.norm2(self.pointwise(x)))


class SelectiveDualPathBiFPN(nn.Module):
    """Frozen five-node graph with global or view-conditioned positive weights."""

    def __init__(self, channels=(64, 64, 64, 64), out_channels=64, view_conditioned=False):
        super().__init__()
        if len(channels) != 4:
            raise ValueError('Expected channels for F88/F44/F22/F11')
        self.view_conditioned = bool(view_conditioned)
        self.align = nn.ModuleDict({
            name: AlignProjection(in_ch, out_channels)
            for name, in_ch in zip(('f88', 'f44', 'f22', 'f11'), channels)
        })
        self.nodes = nn.ModuleDict({name: SeparableNode(out_channels) for name in NODE_SIZES})
        self.raw_weights = nn.ParameterDict()
        for name, count in NODE_SIZES.items():
            positive = [4.0, 1.0] if name == 't22' else [1.0] * count
            self.raw_weights[name] = nn.Parameter(torch.tensor([_inverse_softplus(x) for x in positive]))
        self.view_bias = nn.Linear(32, 12, bias=True) if self.view_conditioned else None
        if self.view_bias is not None:
            nn.init.zeros_(self.view_bias.weight)
            nn.init.zeros_(self.view_bias.bias)
        self.residual = nn.ModuleDict({
            name: nn.Conv2d(out_channels, out_channels, 1, bias=True)
            for name in ('z88', 'z44', 'z22')
        })
        for projection in self.residual.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.last_shapes = {}
        self.last_weights = {}
        self.last_residual_ratios = {}
        self.last_residual_norms = {}

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def _down(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        result = self.pool(x)
        if result.shape[-2:] != ref.shape[-2:]:
            raise RuntimeError(f'Downsample shape {result.shape[-2:]} != reference {ref.shape[-2:]}')
        return result

    def _biases(self, hidden: torch.Tensor | None):
        if self.view_bias is None:
            return {name: None for name in NODE_SIZES}
        if hidden is None or hidden.ndim != 2 or hidden.shape[1] != 32:
            raise ValueError('P2 requires the existing [B,32] Golden C view hidden embedding')
        flat = self.view_bias(hidden)
        return dict(zip(NODE_SIZES, flat.split(tuple(NODE_SIZES.values()), dim=1)))

    def normalized_weights(self, hidden: torch.Tensor | None = None):
        biases = self._biases(hidden)
        result = {}
        for name, raw in self.raw_weights.items():
            effective = raw[None, :] if biases[name] is not None else raw
            if biases[name] is not None:
                effective = effective + biases[name]
            positive = F.softplus(effective)
            result[name] = positive / (positive.sum(dim=-1, keepdim=True) + 1e-4)
        return result

    @staticmethod
    def _fuse(inputs, weights):
        if weights.ndim == 1:
            return sum(w * x for w, x in zip(weights, inputs))
        return sum(weights[:, i, None, None, None] * x for i, x in enumerate(inputs))

    def forward(self, features, hidden: torch.Tensor | None = None):
        f88, f44, f22, f11 = (self.align[n](x) for n, x in zip(('f88', 'f44', 'f22', 'f11'), features))
        weights = self.normalized_weights(hidden)
        t22 = self.nodes['t22'](self._fuse((f22, self._up(f11, f22)), weights['t22']))
        t44 = self.nodes['t44'](self._fuse((f44, self._up(t22, f44)), weights['t44']))
        t88 = self.nodes['t88'](self._fuse((f88, self._up(t44, f88)), weights['t88']))
        b44 = self.nodes['b44'](self._fuse((f44, t44, self._down(t88, f44)), weights['b44']))
        b22 = self.nodes['b22'](self._fuse((f22, t22, self._down(b44, f22)), weights['b22']))
        self.last_shapes = {k: tuple(v.shape) for k, v in {
            'F88': f88, 'F44': f44, 'F22': f22, 'F11': f11,
            'T22': t22, 'T44': t44, 'T88': t88, 'B44': b44, 'B22': b22,
        }.items()}
        self.last_weights = {k: v.detach() for k, v in weights.items()}
        d88 = self.residual['z88'](t88)
        d44 = self.residual['z44'](b44)
        d22 = self.residual['z22'](b22)
        deltas = {'88': d88, '44': d44, '22': d22}
        bases = {'88': features[0], '44': features[1], '22': features[2]}
        self.last_residual_norms = {name: float(value.detach().float().norm()) for name, value in deltas.items()}
        self.last_residual_ratios = {
            name: float(deltas[name].detach().float().norm() / bases[name].detach().float().norm().clamp_min(1e-12))
            for name in deltas
        }
        return (
            features[0] + d88,
            features[1] + d44,
            features[2] + d22,
            features[3],
        )

    def flat_normalized_weights(self, hidden: torch.Tensor | None = None):
        values = self.normalized_weights(hidden)
        return torch.cat([values[name] for name in NODE_SIZES], dim=-1)
