from typing import Optional, Tuple, List

import torch
import torchvision
from torchvision import datasets
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import shutil
import PIL
from zipfile import ZipFile
from tqdm import tqdm

from torchaudio._internal import download_url_to_file

#COLOR_CIR = ['red', 'green', 'yellow', 'blue', 'orange', 'black', 'pink']
COLOR_CIR = None

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def de_normalized(img: torch.Tensor, 
                  mean: List[int], 
                  std: List[int]):
    def f_de_normalized(array: torch.Tensor,
                        mean: int,
                        std: int):
        return array*std + mean
    de_normalized_img = torch.zeros_like(img)
    for i in range(3):
        de_normalized_img[i] = f_de_normalized(img[i], mean[i], std[i])
        de_normalized_img[i] /= de_normalized_img[i].max()
    return (de_normalized_img*255).type(torch.uint8)

def voc_visualize(imgs: list, 
                  save_img: Optional[bool] = False, 
                  saved_img_path: Optional[str] = None) -> None:
    """
    Visulualizing samples from Pacal VOC Object Detection dataset

    Args:
        imgs: VOC-object type
        save_img: Optional[bool]: Allow to whether save or not your figure
        saved_img_path: Optional[str]: Where your saved figure will be in
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, ax = plt.subplots(ncols=len(imgs), squeeze = False)
    for idx, img in enumerate(imgs):
        pil_object = img[0]
        objects = img[1]['annotation']['object']
        image = v2.functional.pil_to_tensor(pil_object)
        voc_bbox = lambda x: [int(x['xmin']), int(x['ymin']), int(x['xmax']), int(x['ymax'])]
        bboxes = []
        names = []
        for oj in objects:
            names.append(oj['name'])
            bboxes.append(voc_bbox(oj['bndbox']))
        bboxes = torch.Tensor(bboxes)
        drawn_boxes = draw_bounding_boxes(image, boxes = bboxes, labels = names, colors = COLOR_CIR)
        ax[0, idx].imshow(drawn_boxes.permute(1, 2, 0))
        ax[0, idx].axis('off')
    if save_img:
        fig.savefig(saved_img_path, bbox_inches = 'tight')
        print("Already saved your image")

def visualize(imgs: List[torch.Tensor], 
              bboxes: List[torch.Tensor],
              names: Optional[List[list]] = None,
              save_img: Optional[bool] = False, 
              saved_img_path: Optional[str] = None,
              *args, **kwargs) -> None:
    """
    Visulualizing images with it's bounding boxes

    Args:
        img: torch.Tensor
        bboxes: torch.Tensor: Bounding boxes coordinates
        names: Optional[list]: Object names will be used for annotation
        save_img: Optional[bool]: Allow to whether save or not your figure
        saved_img_path: Optional[str]: Where your saved figure will be in
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
        bboxes = [bboxes]
        names = [names]
    fig, ax = plt.subplots(ncols=len(imgs), squeeze = False)
    for idx, (img, bbox, name) in enumerate(zip(imgs, bboxes, names)):
        if img.dtype == torch.float32:
            img = de_normalized(img, *args, **kwargs)
            print(img.dtype)
        drawn_boxes = draw_bounding_boxes(img, boxes = bbox, labels = name, colors = COLOR_CIR)
        ax[0, idx].imshow(drawn_boxes.permute(1, 2, 0))
        ax[0, idx].axis('off')
    if save_img:
        fig.savefig(saved_img_path, bbox_inches = 'tight')
        print("Already saved your image")

import zipfile

def _download(data_url: str, 
              dest_directory: str) -> None:
    """
    Download data files from given data url.

    Args:
    data_url: str
    dest_directory: str
        where your files will be at.
    """
    filename = os.path.split(data_url)[-1]

    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filepath = os.path.join(dest_directory, filename)

    download_url_to_file(data_url, filepath)
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_directory)

