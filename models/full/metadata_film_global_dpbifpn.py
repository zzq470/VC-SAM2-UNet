"""P1: Golden C Metadata-FiLM with global selective DPBiFPN."""
import torch.nn as nn
import torch.nn.functional as F

from models.full.SAM2UNet import SAM2UNet as GoldenCSAM2UNet
from models.full.selective_dpbifpn_common import SelectiveDualPathBiFPN


class MetadataFiLMGlobalDPBiFPN(GoldenCSAM2UNet):
    def __init__(self, checkpoint_path=None, view_embedding_dim=16):
        super().__init__(checkpoint_path, view_embedding_dim)
        self.selective_dpbifpn = SelectiveDualPathBiFPN(view_conditioned=False)

    def forward(self, x, view_id):
        x1, x2, x3, x4 = self.encoder(x)
        features = (self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4))
        features = self.view_conditioner(features, view_id)
        x1, x2, x3, x4 = self.selective_dpbifpn(features)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


class D10MetadataFiLMGlobalDPBiFPN(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.base = MetadataFiLMGlobalDPBiFPN(pretrained)
        self.base.side1 = nn.Conv2d(64, 14, 1)
        self.base.side2 = nn.Conv2d(64, 14, 1)
        self.base.head = nn.Conv2d(64, 14, 1)

    def forward(self, x, view_id):
        main, aux1, aux2 = self.base(x, view_id)
        return main, [aux1, aux2]

