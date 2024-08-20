from typing import Tuple

import torch
from torch import nn
import torchvision

class Grid:
    """
    Return the upper-left coordinates of each grid-cell
    """
    def __init__(self,
                 grid_size: Tuple[int, int]):
        self.grid_size = grid_size

    def __call__(self,
                 image: torch.Tensor):
        """
        image: torch.Tensor, shape: (N, *, H, W)
        """
        img_h, img_w = image.shape[-2:]
        grid_cell_h = img_h/self.S_h
        grid_cell_w = img_w/self.S_w
        return grid_cell_h, grid_cell_w

class OffsetCoder:
    def __init__(self, image_size: Tuple[int]):
        self.image_size = image_size

    def decode(self,
               encoded_offsets: torch.Tensor,
               gridcells: torch.Tensor,
               grid_cell_h: float, grid_cell_w: float):
        """
        Decode encoded_offsets and gridcells coordinates to get bounding boxes with the type cxcywh.
        """
        bboxes = torch.zeros_like(encoded_offsets, device = encoded_offsets.device)
        bboxes[:, 0] = encoded_offsets[:, 0]*grid_cell_w + gridcells[:, 0]
        bboxes[:, 1] = encoded_offsets[:, 1]*grid_cell_h + gridcells[:, 1]
        bboxes[:, 2] = encoded_offsets[:, 2]*self.image_size[1]
        bboxes[:, 3] = encoded_offsets[:, 3]*self.image_size[0]
        return bboxes

    def encode(self,
               boxes: torch.Tensor,
               gridcells: torch.Tensor,
               grid_cell_h: float, grid_cell_w: float):
        """
        Encode boxes given gridcells coordinates.
        """
        bboxes = torch.zeros_like(boxes, device = boxes.device)
        bboxes[:, 0] = (boxes[:, 0]-gridcells[:, 0])/grid_cell_w
        bboxes[:, 1] = (boxes[:, 1]-gridcells[:, 1])/grid_cell_h
        bboxes[:, 2] = boxes[:, 2]/self.image_size[1]
        bboxes[:, 3] = boxes[:, 3]/self.image_size[0]
        return bboxes

def box_iou(boxes1: torch.Tensor,
            boxes2: torch.Tensor,
            box_type: str):
    if box_type == 'cxcywh':
            boxes1 = torchvision.ops.box_convert(boxes1, 'cxcywh', 'xyxy')
            boxes2 = torchvision.ops.box_convert(boxes2, 'cxcywh', 'xyxy')
    return torchvision.ops.box_iou(boxes1, boxes2)

class Resnet18(nn.Module):
    def __init__(self,
                 pretrained: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = torchvision.models.resnet18(pretrained)
        delattr(self.net, "avgpool")
        delattr(self.net, "fc")
    def forward(self, X: torch.Tensor):
        return self.net.layer4(self.net.layer3(self.net.layer2(self.net.layer1(self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(X))))))))