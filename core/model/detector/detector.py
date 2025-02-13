from torch import nn

from core.config.config import configurable
from core.model.build import MODEL_REGISTRY, build_model
from core.structures.instances import Instances


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03


@MODEL_REGISTRY.register()
class SpikeYOLO(nn.Module):
    @configurable
    def __init__(self, backbone, neck, head, *args, **kwargs):
        super(SpikeYOLO, self).__init__()
        self.backbone = build_model(backbone, name=backbone.NAME, *args, **kwargs)
        self.neck = build_model(neck, name=neck.NAME, *args, **kwargs)
        self.head = build_model(head, name=head.NAME, ch=self.neck.out_channels, *args, **kwargs)
        
        self.init_weights()

    @classmethod
    def from_config(cls, cfg):
        backbone_cfg = cfg.BACKBONE.EVENT
        neck_cfg = cfg.NECK.EVENT
        head_cfg = cfg.HEAD
        spike_cfg = cfg.SPIKE_CONFIG
        scales = cfg.SCALES
        num_classes = cfg.NUM_CLASSES
        spike_t = cfg.SPIKE_T

        return {
            "backbone": backbone_cfg,
            "neck": neck_cfg,
            "head": head_cfg,
            "spike_cfg": spike_cfg,
            "scales": scales,
            "num_classes": num_classes,
            "spike_t": spike_t,
        }

    def forward(self, events, image=None):
        # 单event通路不需要image输入
        features = self.backbone(events)
        features = self.neck(features)

        return features

    def loss(self, inputs):
        events = inputs["events"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(events, image=None)
        loss_dict = self.head.loss(features, target)

        return loss_dict

    def predict(self, inputs):
        image = inputs["image"]
        events = inputs["events"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(events, image)
        return self.head.predict(features, target)

    def init_weights(self):
        initialize_weights(self)


@MODEL_REGISTRY.register()
class YOLOv8(nn.Module):
    @configurable
    def __init__(self, backbone, neck, head, *args, **kwargs):
        super(YOLOv8, self).__init__()
        self.backbone = build_model(backbone, name=backbone.NAME, *args, **kwargs)
        self.neck = build_model(neck, name=neck.NAME, *args, **kwargs)
        self.head = build_model(head, name=head.NAME, *args, ch=self.neck.out_channels, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        backbone_cfg = cfg.BACKBONE.RGB
        neck_cfg = cfg.NECK.RGB
        head_cfg = cfg.HEAD
        scales = cfg.SCALES
        num_classes = cfg.NUM_CLASSES

        return {
            "backbone": backbone_cfg,
            "neck": neck_cfg,
            "head": head_cfg,
            "scales": scales,
            "num_classes": num_classes,
        }

    def forward(self, image, events=None):
        # 单img通路不需要event输入
        features = self.backbone(image)
        features = self.neck(features)

        return features

    def loss(self, inputs):
        image = inputs["image"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(image, events=None)
        loss_dict = self.head.loss(features, target)

        return loss_dict

    def predict(self, inputs):
        image = inputs["image"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(image, events=None)
        return self.head.predict(features, target)


@MODEL_REGISTRY.register()
class FusionYOLO(nn.Module):
    @configurable
    def __init__(
        self, event_backbone, rgb_backbone, event_neck, rgb_neck, fusion, head, *args, **kwargs
    ):
        super(SpikeYOLO, self).__init__()
        self.event_backbone = build_model(event_backbone, name=event_backbone.NAME, *args, **kwargs)
        self.rgb_backbone = build_model(rgb_backbone, name=rgb_backbone.NAME, *args, **kwargs)
        self.event_neck = build_model(event_neck, name=event_neck.NAME, *args, **kwargs)
        self.rgb_neck = build_model(rgb_neck, name=rgb_neck.NAME, *args, **kwargs)

        self.fusion_module = build_model(fusion, name=fusion.NAME, *args, **kwargs)
        self.head = build_model(head, name=head.NAME, ch=self.neck.out_channels, *args, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        event_backbone_cfg = cfg.BACKBONE.EVENT
        rgb_backbone_cfg = cfg.BACKBONE.RGB
        event_neck_cfg = cfg.NECK.EVENT
        rgb_neck_cfg = cfg.NECK.RGB
        fusion_cfg = cfg.FUSION
        head_cfg = cfg.HEAD
        spike_cfg = cfg.SPIKE_CONFIG
        scales = cfg.SCALES
        num_classes = cfg.NUM_CLASSES
        spike_t = cfg.SPIKE_T

        return {
            "event_backbone": event_backbone_cfg,
            "rgb_backbone": rgb_backbone_cfg,
            "event_neck": event_neck_cfg,
            "rgb_neck": rgb_neck_cfg,
            "fusion": fusion_cfg,
            "head": head_cfg,
            "spike_cfg": spike_cfg,
            "scales": scales,
            "num_classes": num_classes,
            "spike_t": spike_t,
        }

    def forward(self, events, image):
        event_features = self.event_backbone(events=events)
        rgb_features = self.rgb_backbone(image=image)

        event_features = self.event_neck(event_features)
        rgb_features = self.rgb_neck(rgb_features)
        features = self.fusion_module(event_features, rgb_features)

        return features

    def loss(self, inputs):
        events = inputs["events"]
        image = inputs["image"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(events, image)
        loss_dict = self.head.loss(features, target)

        return loss_dict

    def predict(self, inputs):
        events = inputs["events"]
        image = inputs["image"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(events, image)
        return self.head.predict(features, target)
