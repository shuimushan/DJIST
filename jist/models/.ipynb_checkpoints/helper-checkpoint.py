import numpy as np
from . import backbones
from torch import nn


def get_backbone(backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,
                 out_indices=[8, 9, 10, 11]):

    if 'dino' in backbone_arch.lower():
        return backbones.DinoV2_self(model_name=backbone_arch, layer1=layer1,  use_cls=use_cls, norm_descs=norm_descs, out_indices=out_indices)
    else:
        print("wrong input backbone type")
        exit()


