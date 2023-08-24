# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor

from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class DIEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        #print(inputs.shape)  # torch.Size([8, 6, 256, 256])
        #print(f"Backbone_inchannels {self.backbone_inchannels}")
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        #print(img_from.shape)  # torch.Size([8, 3, 256, 256])
        #print(img_to.shape)  # torch.Size([8, 3, 256, 256])
        x = self.backbone(img_from, img_to)
        if self.with_neck:
            x = self.neck(x)
        return x
