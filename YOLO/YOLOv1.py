from typing import Optional, Callable, Tuple, List

import torch
from torch import nn
import torchvision
from torchvision.transforms import v2

import os
import PIL
from PIL import Image

from OD_datasets import VOCDetection, OD_transform, VOC_I2N
from OD_utils import visualize
from YOLOv1_detection import *
import S4T as S

class PascalVOC2012(S.SDataModule):
    def __init__(self,
                 root: str,
                 download: bool,
                 train_transforms: Optional[Callable],
                 test_transforms: Optional[Callable],
                 batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = VOCDetection(root, year = '2012', image_set = 'train', download = download, transforms = train_transforms)
        self.val_dataset = VOCDetection(root, year = '2012', image_set = 'val', download = download, transforms = test_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           collate_fn = self.collate_fn,
                                           num_workers = 4,
                                           prefetch_factor = 1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = self.collate_fn,
                                           num_workers = 4,
                                           prefetch_factor = 1)

    def collate_fn(self, batch):
        img_batch, tar_batch = [], []
        for img, tar in batch:
            img_batch.append(img)
            tar_batch.append(tar)
        return torch.stack(img_batch), tar_batch

class MyYOLO_ResNet18(S.SModule):
    def __init__(self,
                 grid_size: Tuple[int, int],
                 num_classes: int,
                 num_boxes_per_gridcell: int,
                 lr: float):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes_per_gridcell = num_boxes_per_gridcell
        self.lr = lr
        self.backbone = Resnet18(True)
        self.grid = Grid(grid_size)
        self.box_coder = OffsetCoder(image_size = [500, 500])
        self.predictor = nn.Conv2d(512, num_boxes_per_gridcell*5 + num_classes,
                                   kernel_size = 3, padding = 1)

    def forward(self, X: torch.Tensor):
        out = self.backbone(X)
        out = self.predictor(out).permute(0, 2, 3, 1)
        return torch.sigmoid(out)

    def loss(self,
             preds: torch.Tensor,
             targets: List[dict]):
        grid_cell_h, grid_cell_w = self.box_coder.image_size[0]/self.grid.grid_size[0], \
                                   self.box_coder.image_size[1]/self.grid.grid_size[1]

        num_foreground = 0
        local_loss = 0
        obj_conf_loss = 0
        noobj_conf_loss = 0
        prob_loss = 0
        for preds_per_img, targets_per_img in zip(preds, targets):                                   # (S, S, (B*5+C)) vs (m, )
            if targets_per_img['boxes'].numel() == 0:
                continue
            shuffle_idxs = torch.randperm(targets_per_img['boxes'].shape[0])
            gt_bboxes_per_img = torchvision.ops.box_convert(targets_per_img['boxes'], 'xyxy', 'cxcywh')[shuffle_idxs]                         # (m, 4)
            gt_logits_per_img = nn.functional.one_hot(targets_per_img['labels']-1, self.num_classes).type(torch.float)[shuffle_idxs]        # (m, C)

            foreground_gc_idxs_per_img = gt_bboxes_per_img[:, :2].clone()
            foreground_gc_idxs_per_img[:, 0] = foreground_gc_idxs_per_img[:, 0]//grid_cell_w
            foreground_gc_idxs_per_img[:, 1] = foreground_gc_idxs_per_img[:, 1]//grid_cell_h
            foreground_gc_idxs_per_img = foreground_gc_idxs_per_img.type(torch.long)

            gc_idxs_per_img = torch.full(preds_per_img.shape[:2], -1,
                                          dtype = torch.long,
                                          device = preds.device)
            gc_idxs_per_img[foreground_gc_idxs_per_img[:, 1], foreground_gc_idxs_per_img[:, 0]] = torch.arange(start = 0,
                                                                                                               end = len(gt_bboxes_per_img),
                                                                                                               dtype = torch.long,
                                                                                                               device = preds.device)
            gt_matched_idxes_per_img = gc_idxs_per_img[gc_idxs_per_img >= 0] # (objs, )
            num_foreground += len(gt_matched_idxes_per_img)
            foreground_preds_per_img = preds_per_img[gc_idxs_per_img >= 0]   # (objs, (B*5+C))
            background_preds_per_img = preds_per_img[gc_idxs_per_img < 0]    # (noobjs, (B*5+C))

            # noobjs
            noobj_conf_loss += (background_preds_per_img[:, :self.num_boxes_per_gridcell].reshape(-1)**2).sum()

            # objs
            conf_obj_preds_per_img = foreground_preds_per_img[:, :self.num_boxes_per_gridcell]                  # (objs, B)
            offset_preds_per_img = foreground_preds_per_img[:, self.num_boxes_per_gridcell:-self.num_classes]\
                                    .reshape(-1, self.num_boxes_per_gridcell, 4)                                # (objs, B, 4)
            prob_preds_per_img = foreground_preds_per_img[:, -self.num_classes:]                                # (objs, C)
            ## prob_loss
            prob_loss += nn.functional.mse_loss(prob_preds_per_img,
                                                gt_logits_per_img[gt_matched_idxes_per_img], reduction = 'sum')

            # predict_iou
            max_iou_box_idxs = []
            iou_preds = []
            for foreground_offsets, gt_box, upper_lefts in zip(offset_preds_per_img,
                                                               gt_bboxes_per_img[gt_matched_idxes_per_img],
                                                               foreground_gc_idxs_per_img[gt_matched_idxes_per_img]):
                upper_lefts = upper_lefts*torch.tensor([grid_cell_w, grid_cell_h],
                                                       dtype = preds.dtype,
                                                       device = preds.device).unsqueeze(0).repeat_interleave(self.num_boxes_per_gridcell, dim  = 0)             #
                bbox_preds = self.box_coder.decode(foreground_offsets, upper_lefts, grid_cell_h, grid_cell_w)
                iou_pred, max_idx = box_iou(bbox_preds, gt_box.unsqueeze(0), box_type = 'cxcywh').max(dim = 0)
                iou_preds.append(iou_pred)
                max_iou_box_idxs.append(max_idx)
            ## local_loss
            offset_preds_per_img = offset_preds_per_img[torch.arange(0, len(max_iou_box_idxs)), max_iou_box_idxs, :].reshape(-1, 4)
            gt_offsets_per_img = self.box_coder.encode(gt_bboxes_per_img[gt_matched_idxes_per_img],
                                                       foreground_gc_idxs_per_img[gt_matched_idxes_per_img]*\
                                                       torch.tensor([grid_cell_w, grid_cell_h],
                                                       dtype = preds.dtype,
                                                       device = preds.device),
                                                       grid_cell_h, grid_cell_w)

            local_loss += nn.functional.mse_loss(offset_preds_per_img[:, :2], gt_offsets_per_img[:, :2], reduction = 'sum')\
                        + nn.functional.mse_loss(torch.sqrt(offset_preds_per_img[:, 2:]), torch.sqrt(gt_offsets_per_img[:, 2:]), reduction = 'sum')
            ## obj_conf_loss
            obj_conf_loss += nn.functional.mse_loss(conf_obj_preds_per_img[torch.arange(0, len(max_iou_box_idxs)), max_iou_box_idxs].reshape(-1),
                                                    torch.Tensor(iou_preds).to(preds), reduction = 'sum')

        N = max(1, num_foreground)
        return local_loss/N, prob_loss/N, obj_conf_loss/N, noobj_conf_loss/N, sum(iou_preds)/len(iou_preds)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        local_loss, prob_loss, obj_conf_loss, noobj_conf_loss, iou_preds = self.loss(preds, y)
        loss = 5*local_loss + prob_loss + obj_conf_loss + 0.5*noobj_conf_loss
        values = {"train_loss": loss,
                  "train_local_los": local_loss,
                  "train_prob_loss": prob_loss,
                  "train_obj_conf_loss": obj_conf_loss,
                  "train_noobj_conf_loss": noobj_conf_loss,
                  'train_avg_iou_preds': iou_preds}
        self.log_dict(values, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        local_loss, prob_loss, obj_conf_loss, noobj_conf_loss, iou_preds = self.loss(preds, y)
        loss = 5*local_loss + prob_loss + obj_conf_loss + 0.5*noobj_conf_loss
        values = {"val_loss": loss,
                  "val_local_los": local_loss,
                  "val_prob_loss": prob_loss,
                  "val_obj_conf_loss": obj_conf_loss,
                  "val_noobj_conf_loss": noobj_conf_loss,
                  'val_avg_iou_preds': iou_preds}
        self.log_dict(values, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 0.00005)
        return optimizer

def predict(model: nn.Module,
            X: torch.Tensor,
            obj_conf_thres: float,
            cls_conf_thres: float,
            nms_thres: float):
    grid_cell_h, grid_cell_w = model.box_coder.image_size[0]/model.grid.grid_size[0], \
                               model.box_coder.image_size[1]/model.grid.grid_size[1]
    model.eval()
    preds = model(X.unsqueeze(0))
    ## Split general prediction to different specific predictions
    conf_preds = preds[:, :, :, :2]
    offset_preds = preds[:, :, :, 2:10].reshape(1, 16, 16, 2, 4)
    prob_preds = preds[:, :, :, 10:]

    max_conf_preds, max_conf_pred_idxs = conf_preds.max(dim = -1)
    grid_cell_idxs = torch.where(max_conf_preds > obj_conf_thres)

    ## Get reliable predictions
    reliable_class_probs = prob_preds[grid_cell_idxs]
    max_reliable_class_probs, max_class_idxs = reliable_class_probs.max(dim = -1)
    reliable_offsets = offset_preds[grid_cell_idxs][torch.arange(0, len(max_reliable_class_probs)), max_conf_pred_idxs[max_conf_preds > obj_conf_thres], :]
    
    upper_lefts = torch.stack([grid_cell_idxs[2], grid_cell_idxs[1]]).T*torch.tensor([grid_cell_w, grid_cell_h])
    reliable_predicted_boxes = torchvision.ops.box_convert(model.box_coder.decode(reliable_offsets, upper_lefts, grid_cell_h, grid_cell_w), 'cxcywh', 'xyxy')
    
    ##
    confidence_scores = max_reliable_class_probs*max_conf_preds[grid_cell_idxs]
    kept_box_idxs = torchvision.ops.nms(reliable_predicted_boxes, confidence_scores, 
                                        iou_threshold = nms_thres)
    confidence_scores = confidence_scores[kept_box_idxs]
    predicted_classes = max_class_idxs[kept_box_idxs][confidence_scores > cls_conf_thres]
    predicted_boxes = reliable_predicted_boxes[kept_box_idxs][confidence_scores > cls_conf_thres]
    confidence_scores = confidence_scores[confidence_scores > cls_conf_thres]
    return {'class_idxs': predicted_classes,
            'confidence_scores': confidence_scores,
            'predicted_boxes': predicted_boxes}

if __name__ == '__main__':
    # data augmentation
    train_augs = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomPhotometricDistort(p = 0.1),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_augs = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = OD_transform(500, None, None, train_augs)
    test_transform = OD_transform(500, None, None, test_augs)
    
    # dataloader
    data = PascalVOC2012('./VOC',
                         download = True,
                         train_transforms = train_transform,
                         test_transforms = test_transform,
                         batch_size = 32)
    
    checkpoint_callback = S.ModelCheckpoint(dirpath = '/kaggle/working',
                                            save_top_k = 7, monitor = 'val_avg_iou_preds',
                                            mode = 'max',
                                            filename = 'mYOLOv1-voc-12-epoch:%02d-val_avg_iou_preds:%.4f')

    # setup and train model
    torch.manual_seed(42)
    net = MyYOLO_ResNet18(grid_size = (16, 16),
                          num_classes = 20,
                          num_boxes_per_gridcell = 2,
                          lr = 1e-3)
    trainer = S.Trainer(accelerator="gpu",
                        callbacks = [checkpoint_callback],
                        enable_checkpointing=True,
                        max_epochs = 50)
    trainer.fit(net, data)

    # let's take a try
    val_dataset = data.val_dataset
    preds = predict(net, val_dataset[9][0], 0.15, 0.15, 0.3)
    visualize(val_dataset[9][0], preds['predicted_boxes'], names = [f'{VOC_I2N[cls_idx.item()+1]}_{conf_score:.2f}' for (cls_idx, conf_score) in zip(preds['class_idxs'], preds['confidence_scores'])], mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])    
