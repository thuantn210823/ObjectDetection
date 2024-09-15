from typing import Optional, Callable, List

import torch
from torch import nn
import torchvision
from torchvision.transforms import v2

from OD_datasets import VOCDetection, OD_transform
from SSD_detection import AnchorGenerator, Matcher, OffsetCoder
import S4T as S

class PascalVOC2012(S.SDataModule):
    """
    Pascal VOC 2012 dataset.

    Parameters:
    root: str - Root directory of the VOC Dataset.
    download: bool - Whether to download dataset or not
    train_transforms: Optional[Callable] - Data augmentations for train data.
    test_transforms: Optional[Callable] - Data augmentations for test data.
    batch_size: int - The number of samples per batch.
    """
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

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors*(num_classes + 1), kernel_size = 3, padding = 1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors*4, kernel_size = 3, padding = 1)

class SimpleSSD(S.SModule):
    """
    My simple SSD model. I used MobileNetv3 as the model's backbone.

    Parameters:
    num_classes: int - The number of labels.
    iou_thresh: float - The threshold value for assgining anchors. Default = 0.5.
    positive_fraction: float - The number of positive samples per batch which later used for hard mining.
    lr: float - The learning rate value.
    """
    def __init__(self,
                 num_classes: int,
                 iou_thresh: float = 0.5,
                 positive_fraction: float = 0.25,
                 lr: float = 0.5,
                 *args, **kwargs
                 ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        # Configure a simple ssd model 
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained = True).features
        self.blk_idx = [0, 2, 7, 12]
        channel_idx = [16, 24, 48, 576]
        self.anchor_generator = {
            0: AnchorGenerator(sizes = [0.2, 0.272],
                                   ratios = [0.5, 1, 2]),
            2: AnchorGenerator(sizes = [0.37, 0.447],
                                   ratios = [0.5, 1, 2]),
            7: AnchorGenerator(sizes = [0.54, 0.619],
                                   ratios = [0.5, 1, 2]),
            12: AnchorGenerator(sizes = [0.71, 0.79],
                                   ratios = [0.5, 1, 2])
        }
        num_anchors = len(self.anchor_generator[0].sizes) + len(self.anchor_generator[0].ratios) - 1

        for i in range(len(channel_idx)):
            setattr(self, f'cls_{self.blk_idx[i]}', cls_predictor(channel_idx[i],
                                                    num_anchors,
                                                    num_classes))
            setattr(self, f'bbox_{self.blk_idx[i]}', bbox_predictor(channel_idx[i],
                                                                    num_anchors))

        self.box_coder = OffsetCoder(weights = (10.0, 10.0, 5.0, 5.0))
        self.proposal_matcher = Matcher(iou_thresh)
        self.neg_to_pos_ratio = (1.0 - positive_fraction)/positive_fraction

    def forward(self,
                images: torch.Tensor):
        """
        Parameters:
        images: Tensor - The input images.

        Returns: 
        Tuple of the following items;
        Tensor: Anchor boxes.
        Tensor: Class predictions.
        Tensor: Bounding box predictions.
        """
        X = images
        cls_preds = []
        bbox_regression_preds = []
        anchors = []
        for i in range(len(self.backbone)):
            X = self.backbone[i](X)
            if i in self.blk_idx:
                cls_preds.append(getattr(self, f'cls_{i}')(X).permute(0, 2, 3, 1).flatten(start_dim = 1))
                bbox_regression_preds.append(getattr(self, f'bbox_{i}')(X).permute(0, 2, 3, 1).flatten(start_dim = 1))
                anchors.append(self.anchor_generator[i](X).unsqueeze(0).repeat_interleave(repeats = X.shape[0], dim = 0).flatten(start_dim = 1))
        cls_preds = torch.cat(cls_preds, dim = 1).reshape(X.shape[0], -1, self.num_classes + 1)
        bbox_regression_preds = torch.cat(bbox_regression_preds, dim = 1).reshape(X.shape[0], -1, 4)
        anchors = torch.cat(anchors, dim = 1).reshape(X.shape[0], -1, 4).to(X)*500
        return anchors, cls_preds, bbox_regression_preds

    def loss(self,
             anchors: torch.Tensor,
             cls_preds: torch.Tensor,
             bbox_preds: torch.Tensor,
             targets: List[dict]):
        """
        Loss function. In here we do the same procedure for any object detection problem.
        - Assign target bounding boxes to anchors.
        - Calcuate classification loss and localization loss.
        """
        # Step 1: Assign label for anchors
        matched_idxs = []
        for anchors_per_img, targets_per_img in zip(anchors, targets):
            if targets_per_img['boxes'].numel() == 0:
                matched_idxs.append(torch.full(anchors_per_img.shape[0],), -1,
                                    dtype = torch.int64,
                                    device = anchors_per_img.device)
                continue
            match_quality_matrix = torchvision.ops.box_iou(targets_per_img['boxes'],
                                                           anchors_per_img)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        # Step 2: Calculate Losses
        num_foreground = 0
        bbox_loss = 0
        cls_loss = 0
        for (anchors_per_img,
             cls_preds_per_img,
             bbox_preds_per_img,
             matched_idxs_per_img,
             targets_per_img) in zip(anchors, cls_preds, bbox_preds, matched_idxs, targets):
            foreground_idxs_per_img = torch.where(matched_idxs_per_img>=0)[0]
            foreground_matched_idxs_per_img = matched_idxs_per_img[foreground_idxs_per_img]
            num_foreground_per_img = len(foreground_idxs_per_img)
            num_foreground += num_foreground_per_img

            # Regression
            assigned_anchors_per_img = anchors_per_img[foreground_idxs_per_img]
            gt_bbox_per_img = targets_per_img['boxes'][foreground_matched_idxs_per_img]
            bbox_preds_per_img = bbox_preds_per_img[foreground_idxs_per_img]
            offsets_per_img = self.box_coder.encode(gt_bbox_per_img, assigned_anchors_per_img)
            bbox_loss += torch.nn.functional.smooth_l1_loss(bbox_preds_per_img, offsets_per_img, reduction = 'sum') # Need calculate the number of foregrounds

            # Classification
            gt_cls_per_img = torch.zeros((cls_preds_per_img.shape[0], ),
                                         dtype = targets_per_img['labels'].dtype,
                                         device = targets_per_img['labels'].device)
            gt_cls_per_img[foreground_idxs_per_img] = targets_per_img['labels'][foreground_matched_idxs_per_img]
            background_idxs_per_img = torch.where(matched_idxs_per_img<0)[0]
            negative_ratio = (self.neg_to_pos_ratio*num_foreground_per_img)/len(background_idxs_per_img)
            background_idxs_per_img = background_idxs_per_img[torch.rand(*background_idxs_per_img.shape)<negative_ratio]
            cls_filter = torch.cat([foreground_idxs_per_img, background_idxs_per_img])
            cls_loss += torch.nn.functional.cross_entropy(cls_preds_per_img[cls_filter], gt_cls_per_img[cls_filter], reduction = 'sum')

        N = max(1, num_foreground)
        return bbox_loss/N, cls_loss/N

    def training_step(self, batch, batch_idx):
        x, y = batch
        anchors, cls_preds, bbox_regression_preds = self.forward(x, y)
        bbox_loss, cls_loss = self.loss(anchors, cls_preds, bbox_regression_preds, y)
        loss = bbox_loss + cls_loss
        values = {"train_loss": loss,
                  "train_bbox_los": bbox_loss,
                  "train_cls_loss": cls_loss}
        self.log_dict(values, pbar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        anchors, cls_preds, bbox_regression_preds = self.forward(x, y)
        bbox_loss, cls_loss = self.loss(anchors, cls_preds, bbox_regression_preds, y)
        loss = bbox_loss + cls_loss
        values = {"val_loss": loss,
                  "val_bbox_loss": bbox_loss,
                  "val_cls_loss": cls_loss}
        self.log_dict(values, pbar = True)

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr = self.lr, weight_decay = 0.00005)
        return optimizer

### Predict (Update soon)

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
                                            save_top_k = 7, monitor = 'val_loss',
                                            mode = 'max',
                                            filename = 'mSSD-voc-12-epoch:%02d-val_loss:%.4f')
    # setup and train model
    torch.manual_seed(42)
    net = SimpleSSD(20, lr = 0.0001)

    trainer = S.Trainer(accelerator="gpu",
                        callbacks = [checkpoint_callback],
                        enable_checkpointing=True,
                        max_epochs=50)
    trainer.fit(net, data)