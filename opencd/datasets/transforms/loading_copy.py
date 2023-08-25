# Copyright (c) Open-CD. All rights reserved.
import sys
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from opencd.registry import TRANSFORMS

imgs = []
filename = "/home/agustin/Desktop/Change_Detection/Repositorios/fork-open-cd/data/CHACO/train/T0/00000000000000002785.png"
file_client_args = None
backend_args = None
color_type = "color"
imdecode_backend = "cv2"

try:
    if file_client_args is not None:
        file_client = fileio.FileClient.infer_client(
            file_client_args, filename)
        img_bytes = file_client.get(filename)
    else:
        img_bytes = fileio.get(
            filename, backend_args=backend_args)
    img = mmcv.imfrombytes(
        img_bytes, flag=color_type, backend=imdecode_backend)
    imgs.append(img)
except Exception as e:
    print(e)
img = imgs[0]
print(f"Len imgs: {len(img)}")
print(f"Type {type(img)}")
print(f"Shape {img.shape}")
print(img)
#print(f"Mean Brand 0 {np.mean(img[0])}")
#print(f"Mean Brand 1 {np.mean(img[1])}")
#print(f"Mean Brand 2 {np.mean(img[2])}")

mean = [np.mean(img[0]), np.mean(img[1]), np.mean(img[2])]

std = [np.std(img[0]), np.std(img[1]), np.std(img[2])]

print(f"Mean Array {mean}")

print(f"Std Array {std}")

img = (img - mean) / std

print(f"Img: {img}")