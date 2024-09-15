from typing import List

import torch
from torch import nn
import torchvision

class AnchorGenerator:
    def __init__(self,
                 heights: list,
                 widths: list):
        self.heights = heights
        self.widths = widths
    
    def __call__(self, 
                 upper_lefts: torch.Tensor,
                 grid_cell_h: float,
                 grid_cell_w: float,
                 device: str = 'cpu'):
        """
        Returns: (cxcywh)
        """
        return torch.tensor([[upper_lefts[0] + grid_cell_w/2]*len(self.widths), 
                             [upper_lefts[1] + grid_cell_h/2]*len(self.heights), 
                             self.widths, self.widths], device = device).T

class Matcher:
    def __init__(self, 
                 threshold: int = 0.5):
        self.threshold = threshold
    
    def __call__(self, 
                 match_quality_matrix: torch.Tensor):
        """
        Inputs:
            match_quality_matrix: torch.Tensor. 
                Should have shape (N, M) in there N, M are the number of ground-truth bounding box and anchor boxes respectively

        Outputs:
            Anchors_bbox_map: torch.Tensor: anchors are with no grouth-truth match will be labeled -1
        """
        num_gt_bbox, num_anchors = match_quality_matrix.shape
        jaccard = match_quality_matrix.clone()

        max_iou_anchor_values, anchors_bbox_map = torch.max(jaccard, dim = 0)
        anchors_bbox_map[max_iou_anchor_values<self.threshold] = -1
        for _ in range(num_gt_bbox):
            max_idx = torch.argmax(jaccard)
            anchors_bbox_map[max_idx%num_anchors] = max_idx//num_anchors
            jaccard[:, max_idx%num_anchors] = -1
            jaccard[max_idx//num_anchors, :] = -1
        return anchors_bbox_map

class OffsetCoder:
    def __init__(self, 
                 weights: List[int],
                 eps: float = 1e-6):
        self.weights = weights
        self.eps = eps

    def decode(self,
               encoded_offsets: torch.Tensor,
               anchors: torch.Tensor,
               grid_cell_h: float,
               grid_cell_w: float):
        """
        """
        gridcells = anchors[:, :2] - torch.tensor([grid_cell_w/2, grid_cell_w/2], device = anchors.device)
        bboxes = torch.zeros_like(encoded_offsets, device = anchors.device)
        bboxes[:, 0] = encoded_offsets[:, 0]*grid_cell_w/self.weights[0] + gridcells[:, 0]
        bboxes[:, 1] = encoded_offsets[:, 1]*grid_cell_h/self.weights[1] + gridcells[:, 1]
        bboxes[:, 2] = torch.exp(encoded_offsets[:, 2]/self.weights[2])*anchors[:, 2]
        bboxes[:, 3] = torch.exp(encoded_offsets[:, 3]/self.weights[3])*anchors[:, 3]
        return bboxes

    def encode(self,
               boxes: torch.Tensor,
               anchors: torch.Tensor,
               grid_cell_h: float,
               grid_cell_w: float):
        """
        """
        gridcells = anchors[:, :2] - torch.tensor([grid_cell_w/2, grid_cell_h/2], device = anchors.device)
        bboxes = torch.zeros_like(boxes, device = anchors.device)
        bboxes[:, 0] = self.weights[0]*(boxes[:, 0]-gridcells[:, 0])/grid_cell_w
        bboxes[:, 1] = self.weights[1]*(boxes[:, 1]-gridcells[:, 1])/grid_cell_h
        bboxes[:, 2] = self.weights[2]*(torch.log(boxes[:, 2]/anchors[:, 2] + self.eps))
        bboxes[:, 3] = self.weights[3]*(torch.log(boxes[:, 3]/anchors[:, 3] + self.eps))
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