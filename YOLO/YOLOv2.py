from typing import Optional, Callable, Tuple, List

import torch
from torch import nn
import torchvision
from torchvision.transforms import v2

import os
import PIL
from PIL import Image
import collections

from OD_datasets import VOCDetection, OD_transform, VOC_I2N
from OD_utils import visualize
from YOLOv2_detection import *
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
                 anchor_size: List[list],
                 iou_thres: float,
                 lr: float,
                 image_size: Optional[tuple] = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes_per_gridcell = len(anchor_size[0])
        self.grid_size = grid_size
        self.image_size = image_size
        self.lr = lr
        self.anchor_generator = AnchorGenerator(heights = anchor_size[0],
                                                widths = anchor_size[1])
        self.proposal_matcher = Matcher(iou_thres)
        self.box_coder = OffsetCoder(weights = (1, 1, 1, 1))
        self.backbone = Resnet18(True)
        self.predictor = nn.Conv2d(512, self.num_boxes_per_gridcell*(5 + num_classes),
                                   kernel_size = 3, padding = 1)

    def forward(self, X: torch.Tensor):
        out = self.backbone(X)
        out = self.predictor(out).permute(0, 2, 3, 1)
        return out.reshape(out.shape[0], out.shape[1], out.shape[2], self.num_boxes_per_gridcell, -1)

    def loss(self,
             preds: torch.Tensor,
             targets: List[dict]):
        grid_cell_h, grid_cell_w = self.image_size[0]/self.grid_size[0], \
                                   self.image_size[1]/self.grid_size[1]

        num_foreground = 0
        local_loss = 0
        obj_conf_loss = 0
        noobj_conf_loss = 0
        prob_loss = 0
        total_iou = 0
        for preds_per_img, targets_per_img in zip(preds, targets):                                   # (S, S, B, 5+C) vs (m, )
            if targets_per_img['boxes'].numel() == 0:
                continue
            shuffle_idxs = torch.randperm(targets_per_img['boxes'].shape[0])
            gt_bboxes_per_img = torchvision.ops.box_convert(targets_per_img['boxes'], 'xyxy', 'cxcywh')[shuffle_idxs]                         # (m, 4)
            gt_logits_per_img = nn.functional.one_hot(targets_per_img['labels']-1, self.num_classes).type(torch.float)[shuffle_idxs]          # (m, C)

            foreground_gc_idxs_per_img = gt_bboxes_per_img[:, :2].clone()
            foreground_gc_idxs_per_img[:, 0] = foreground_gc_idxs_per_img[:, 0]//grid_cell_w
            foreground_gc_idxs_per_img[:, 1] = foreground_gc_idxs_per_img[:, 1]//grid_cell_h
            foreground_gc_idxs_per_img = foreground_gc_idxs_per_img.type(torch.long)

            foreground_gc_matched_idxs = collections.defaultdict(list)
            for idx, fg_gc_idx in enumerate(foreground_gc_idxs_per_img):
                foreground_gc_matched_idxs[(fg_gc_idx[1].item(), fg_gc_idx[0].item())].append(idx)
            foreground_gc_matched_idxs = collections.OrderedDict(sorted(foreground_gc_matched_idxs.items()))

            matched_idxs_per_img = torch.full((self.grid_size[0], self.grid_size[1], self.num_boxes_per_gridcell), -1,
                                              dtype = torch.int64, device = preds.device)                     # (S, S, B)
            assigned_anchors_per_img = []
            for gc_idx, matched_per_gc in foreground_gc_matched_idxs.items():
                anchors = self.anchor_generator((gc_idx[1]*grid_cell_w, gc_idx[0]*grid_cell_h),
                                                grid_cell_h,
                                                grid_cell_w,
                                                device = preds.device)
                matched_per_gc = torch.tensor(matched_per_gc, device = preds.device)
                mqmat_per_gc = box_iou(gt_bboxes_per_img[matched_per_gc], anchors, box_type = 'cxcywh')
                fg_idxs_per_gc = self.proposal_matcher(mqmat_per_gc)
                matched_idxs_per_gc = fg_idxs_per_gc[fg_idxs_per_gc>=0]
                matched_idxs_per_img[gc_idx[0], gc_idx[1]][fg_idxs_per_gc>=0] = matched_per_gc[matched_idxs_per_gc]
                assigned_anchors_per_img.append(anchors[fg_idxs_per_gc>=0])
            assigned_anchors_per_img = torch.concat(assigned_anchors_per_img, dim = 0)                         # (objs, 4)

            foreground_preds_per_img = preds_per_img[matched_idxs_per_img >= 0]                                # (objs, 5+C)
            background_preds_per_img = preds_per_img[matched_idxs_per_img < 0]                                 # (noobjs, 5+C)
            gt_matched_idxes_per_img = matched_idxs_per_img[matched_idxs_per_img >= 0]
            num_foreground += len(gt_matched_idxes_per_img)

            # noobjs
            neg_ratio = 3*len(gt_matched_idxes_per_img)/background_preds_per_img.shape[0]
            background_preds_per_img = background_preds_per_img[:, 0].reshape(-1)[torch.rand(background_preds_per_img.shape[0],)<neg_ratio]
            noobj_conf_loss += nn.functional.binary_cross_entropy(torch.sigmoid(background_preds_per_img),
                                                              torch.tensor([0]*len(background_preds_per_img)).to(preds), reduction = 'sum')
            # objs
            conf_obj_preds_per_img = torch.sigmoid(foreground_preds_per_img[:, 0])                              # (objs, 1)
            offset_preds_per_img = foreground_preds_per_img[:, 1:  5]                                           # (objs, 4)
            offset_preds_per_img[:, :2] = torch.sigmoid(offset_preds_per_img[:, :2])
            prob_preds_per_img = foreground_preds_per_img[:, -self.num_classes:]                                # (objs, C)
            ## prob_loss
            prob_loss += nn.functional.cross_entropy(prob_preds_per_img,
                                                     gt_logits_per_img[gt_matched_idxes_per_img], reduction = 'sum')
            # compute iou
            predicted_boxes_per_img = self.box_coder.decode(offset_preds_per_img,
                                                            assigned_anchors_per_img,
                                                            grid_cell_h, grid_cell_w)
            total_iou += torch.diag(box_iou(gt_bboxes_per_img[gt_matched_idxes_per_img],
                                    predicted_boxes_per_img, box_type = 'cxcywh')).sum()
            ## local_loss
            gt_offsets_per_img = self.box_coder.encode(gt_bboxes_per_img[gt_matched_idxes_per_img],
                                                       assigned_anchors_per_img,
                                                       grid_cell_h, grid_cell_w)
            local_loss += nn.functional.smooth_l1_loss(offset_preds_per_img[:, :2], gt_offsets_per_img[:, :2], reduction = 'sum')\
                        + nn.functional.smooth_l1_loss(offset_preds_per_img[:, 2:], gt_offsets_per_img[:, 2:], reduction = 'sum')
            ## obj_conf_loss
            obj_conf_loss += nn.functional.binary_cross_entropy(conf_obj_preds_per_img.reshape(-1),
                                                                torch.Tensor([1]*len(conf_obj_preds_per_img)).to(preds), reduction = 'sum')

        N = max(1, num_foreground)
        return local_loss/N, prob_loss/N, obj_conf_loss/N, noobj_conf_loss/N, total_iou/N

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        local_loss, prob_loss, obj_conf_loss, noobj_conf_loss, avg_iou = self.loss(preds, y)
        loss = 2*local_loss + prob_loss + obj_conf_loss + 0.5*noobj_conf_loss
        values = {"train_loss": loss,
                  "train_local_los": local_loss,
                  "train_prob_loss": prob_loss,
                  "train_obj_conf_loss": obj_conf_loss,
                  "train_noobj_conf_loss": noobj_conf_loss,
                  "train_avg_iou": avg_iou}
        self.log_dict(values, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        local_loss, prob_loss, obj_conf_loss, noobj_conf_loss, avg_iou = self.loss(preds, y)
        loss = 2*local_loss + prob_loss + obj_conf_loss + 0.5*noobj_conf_loss
        values = {"val_loss": loss,
                  "val_local_los": local_loss,
                  "val_prob_loss": prob_loss,
                  "val_obj_conf_loss": obj_conf_loss,
                  "val_noobj_conf_loss": noobj_conf_loss,
                  "val_avg_iou": avg_iou}    
        self.log_dict(values, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 0.00005)
        return optimizer

def predict(model: nn.Module,
            X: torch.Tensor,
            obj_conf_thres: float,
            cls_conf_thres: float,
            nms_thres: float,
            cls_loss_type: Optional[str] = 'CE'):
    """
    cls_loss: "CE"/"ce" or "BCE"/bce. If "CE" was used softmax will be used for inference else sigmoid will be used instead.
    """
    cls_map = {"bce": torch.sigmoid,
               "ce": lambda x: torch.softmax(x, dim = -1)}
    grid_cell_h, grid_cell_w = model.image_size[0]/model.grid_size[0], \
                               model.image_size[1]/model.grid_size[1]
    model.eval()
    preds = model(X.unsqueeze(0))
    ## Split general prediction to different specific prediction
    conf_preds = torch.sigmoid(preds[:, :, :, :, 0])
    offset_preds = preds[:, :, :, :, 1:5]
    offset_preds[:, :, :, :, :2] = torch.sigmoid(offset_preds[:, :, :, :, :2])
    prob_preds = cls_map[cls_loss_type.lower()](preds[:, :, :, :, 5:])
    
    reliable_conf_idxs = torch.where(conf_preds>obj_conf_thres)
    if len(reliable_conf_idxs[0]) == 0:
        pred_clsses, conf_scores, pred_bboxes = [], [], []
    else:
        upper_lefts = torch.cat([reliable_conf_idxs[2].unsqueeze(1), reliable_conf_idxs[1].unsqueeze(1)], dim = 1)*torch.tensor([grid_cell_h, grid_cell_w])
        anchors = torch.cat([model.anchor_generator(upper_left, grid_cell_h, grid_cell_w) for upper_left in upper_lefts], dim = 0).reshape(-1, model.num_boxes_per_gridcell, 4)
        reliable_anchors = anchors[torch.arange(0, anchors.shape[0]), reliable_conf_idxs[-1], :]
        reliable_offsets = offset_preds[reliable_conf_idxs]
        pred_bboxes = torchvision.ops.box_convert(model.box_coder.decode(reliable_offsets, reliable_anchors, grid_cell_h, grid_cell_w), 'cxcywh', 'xyxy')
        max_reliable_class_probs, pred_classes = prob_preds[reliable_conf_idxs].max(dim = -1)
        conf_scores = conf_preds[reliable_conf_idxs]*max_reliable_class_probs
        kept_box_idxs = torchvision.ops.nms(pred_bboxes, conf_scores, iou_threshold = nms_thres)
    
        conf_scores = conf_scores[kept_box_idxs]
        pred_clsses = pred_classes[kept_box_idxs][conf_scores>cls_conf_thres]
        pred_bboxes = pred_bboxes[kept_box_idxs][conf_scores>cls_conf_thres]
        conf_scores = conf_scores[conf_scores>cls_conf_thres]
    return {'class_idxs': pred_clsses,
            'confidence_scores': conf_scores,
            'predicted_boxes': pred_bboxes}

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
                                            filename = 'mYOLOv2-voc-12-epoch:%02d-val_avg_iou_preds:%.4f')

    # setup and train model
    torch.manual_seed(42)
    net = MyYOLO_ResNet18(grid_size = (16, 16),
                          num_classes = 20,
                          anchor_size = [[135.7643, 257.4366, 379.2922, 236.0847,  54.3663],
                                         [127.8012, 173.5446, 302.6718, 414.0100,  43.1354]],
                          iou_thres = 0.6,
                          image_size = (500, 500),
                          lr = 1e-3)
    trainer = S.Trainer(accelerator="gpu",
                        callbacks = [checkpoint_callback],
                        enable_checkpointing=True,
                        max_epochs = 50)
    trainer.fit(net, data)

    # let's take a try
    val_dataset = data.val_dataset
    preds = predict(net, val_dataset[9][0], 0.9, 0.9, 0.4)
    visualize(val_dataset[9][0], preds['predicted_boxes'], names = [f'{VOC_I2N[cls_idx.item()+1]}_{conf_score:.2f}' for (cls_idx, conf_score) in zip(preds['class_idxs'], preds['confidence_scores'])], mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])   

