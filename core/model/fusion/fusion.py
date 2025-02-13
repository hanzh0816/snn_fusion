from torch import nn
import torch

from core.model.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, events_features: list[torch.Tensor], rgb_features):
        for i in range(len(events_features)):
            events_features[i] = events_features[i].mean(dim=0)  # [T,B,...] -> [B,...]
        
