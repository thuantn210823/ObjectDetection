from typing import List, Optional, Callable, Any

import torch
from torch import nn
import torchvision

import copy

class AnchorGenerator(nn.Module):
    def __init__(self, 
                 sizes: List[float], 
                 ratios: List[float],
                 anchor_type: Optional[str] = None) -> None:
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.anchor_type = anchor_type

    def forward(self, 
                feature_maps: torch.Tensor) -> torch.Tensor:
        return self.multibox_prior(feature_maps, self.sizes, self.ratios)

    def multibox_prior(self, 
                       data: torch.Tensor, 
                       sizes: List[float], 
                       ratios: List[float]) -> torch.Tensor:
        """Generate anchor boxes with different shapes centered on each pixel."""
        in_height, in_width = data.shape[-2:]
        device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
        boxes_per_pixel = (num_sizes + num_ratios - 1)
        size_tensor = torch.tensor(sizes, device=device)
        ratio_tensor = torch.tensor(ratios, device=device)
        # Offsets are required to move the anchor to the center of a pixel. Since
        # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height  # Scaled steps in y axis
        steps_w = 1.0 / in_width  # Scaled steps in x axis

        # Generate all center points for the anchor boxes
        center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        # Generate `boxes_per_pixel` number of heights and widths that are later
        # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
        if self.anchor_type == 'square':
            w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                           sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                           * in_height / in_width  # Handle rectangular inputs
        else:
            w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                          sizes[0] * torch.sqrt(ratio_tensor[1:])))
        h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                      sizes[0] / torch.sqrt(ratio_tensor[1:])))
        # Divide by 2 to get half height and half width
        anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

        # Each center point will have `boxes_per_pixel` number of anchor boxes, so
        # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
        out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                    dim=1).repeat_interleave(boxes_per_pixel, dim=0)
        output = out_grid + anchor_manipulations
        return output
    
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
        jaccard = copy.deepcopy(match_quality_matrix)

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
    
    def encode(self,
               assigned_bboxes: torch.Tensor, 
               anchors: torch.Tensor):
        """
        Encode the relation between anchor boxes and their assigned bounding boxes supported for regression part.

        Inputs:
            assigned_bboxes, anchors have the same shape and in the type 'xyxy'. (*, 4)
        Outputs:
            Encoded offsets between the given assigned bboxes and generated anchor boxes.
        """
        encoded_offsets = torch.zeros_like(assigned_bboxes, device = anchors.device)
        converted_anchors = torchvision.ops.box_convert(anchors, 'xyxy', 'cxcywh')
        converted_assigned_bboxes = torchvision.ops.box_convert(assigned_bboxes, 'xyxy', 'cxcywh')
        encoded_offsets[:, 0] = self.weights[0]*(converted_assigned_bboxes[:, 0] - converted_anchors[:, 0])/converted_anchors[:, 2]
        encoded_offsets[:, 1] = self.weights[1]*(converted_assigned_bboxes[:, 1] - converted_anchors[:, 1])/converted_anchors[:, 3]
        encoded_offsets[:, 2] = self.weights[2]*torch.log(converted_assigned_bboxes[:, 2]/converted_anchors[:, 2] + self.eps)
        encoded_offsets[:, 3] = self.weights[3]*torch.log(converted_assigned_bboxes[:, 3]/converted_anchors[:, 3] + self.eps)
        return encoded_offsets

    def decode(self,
               encoded_offsets: torch.Tensor,
               anchors: torch.Tensor):
        """
        Decode encoded_offsets and anchor boxes to get the bounding boxes with type 'xyxy'.
        """
        bboxes = torch.zeros_like(anchors, device = anchors.device)
        converted_anchors = torchvision.ops.box_convert(anchors, 'xyxy', 'cxcywh')
        bboxes[:, 0] = encoded_offsets[:, 0]*converted_anchors[:, 2]/self.weights[0] + converted_anchors[:, 0]
        bboxes[:, 1] = encoded_offsets[:, 1]*converted_anchors[:, 3]/self.weights[1] + converted_anchors[:, 1]
        bboxes[:, 2] = torch.exp(encoded_offsets[:, 2]/self.weights[2])*converted_anchors[:, 2]
        bboxes[:, 3] = torch.exp(encoded_offsets[:, 3]/self.weights[3])*converted_anchors[:, 3]
        # Convert bboxes from 'xywh' to 'xyxy'
        return torchvision.ops.box_convert(bboxes, 'cxcywh', 'xyxy')

