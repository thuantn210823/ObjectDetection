from typing import Optional, Tuple, List, Callable, Any

import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2

import numpy as np

import os
import shutil
import PIL

from OD_utils import _download

COCO_TRAIN_URL = 'http://images.cocodataset.org/zips/train2017.zip'
COCO_VAL_URL = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_TRAINVAL_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


DOWNLOAD_MAP = {
    'train': COCO_TRAIN_URL,
    'val': COCO_VAL_URL
}

VOC_I2N = {1:'aeroplane',
2:'bicycle',
3:'bird',
4:'boat',
5:'bottle',
6:'bus',
7:'car',
8:'cat',
9:'chair',
10:'cow',
11:'diningtable',
12:'dog',
13:'horse',
14:'motorbike',
15:'person',
16:'pottedplant',
17:'sheep',
18:'sofa',
19:'train',
20:'tvmonitor'}
VOC_N2I = {n: i for (i, n) in zip(VOC_I2N.keys(), VOC_I2N.values())}

class OD_transform(v2.Transform):
    def __init__(self,
                 max_size: int,
                 image_mean: List[float],
                 image_std: List[float],
                 f_augs: Optional[Callable] = None) -> None:
        super().__init__()
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.delta = lambda x: max_size - x
        self.f_augs = f_augs

    def forward(self, image: PIL.Image, target: tv_tensors.BoundingBoxes) -> Any:
        w, h = image.size
        w_delta = self.delta(w)
        h_delta = self.delta(h)
        if w_delta > 0:
            image, target = v2.Pad([w_delta//2, 0])(image, target)
            w = self.max_size
        else:
            image, target = v2.CenterCrop([h, self.max_size])(image, target)
        if h_delta > 0:
            image, target = v2.Pad([0, h_delta//2])(image, target)
            h = self.max_size
        else:
            image, target = v2.CenterCrop([self.max_size, h])(image, target)
        image, target = v2.Resize((self.max_size, self.max_size))(image, target)
        return self.f_augs(image, target) if self.f_augs else (image, target)


class COCODetection:
    """

    """
    def __init__(self, 
                 root: str, 
                 download: bool = True,
                 subset: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None
                 ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        if download:
            _download(DOWNLOAD_MAP[subset], root)
            _download(COCO_TRAINVAL_ANN_URL, root)
        _, img_folder = os.path.split(DOWNLOAD_MAP[subset])
        img_folder = os.path.splitext(img_folder)[0]
        self.dataset = torchvision.datasets.CocoDetection(os.path.join(root, img_folder), annFile = os.path.join(root, 'annotations', 'instances_{}2017.json'.format(subset)))

    def __getitem__(self, idx):
        img, ann = self.dataset[idx]
        if self.transform or self.target_transform or self.transforms:
            ids = []
            bboxes = []
            for obj in ann:
                ids.append(obj['category_id'])
                bboxes.append(obj['bbox'])
            bboxes = torch.Tensor(bboxes)
            bboxes = torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy')
            bboxes = tv_tensors.BoundingBoxes(bboxes, format = 'XYXY', canvas_size = img.size)

            if self.transform and self.target_transform:
                img = self.transofrm(img)
                bboxes = self.target_transform(bboxes)
            elif self.transforms:
                img, bboxes = self.transforms(img, bboxes)
            else:
                raise Exception('We need transform for both image and targets')
            return (img, {'image_id': ann[0]['image_id'],'boxes': bboxes, 'labels': ids})
        else:
            return img, ann
    
    def __len__(self):
        return len(self.dataset)
    
class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self,
                 root: str,
                 year: str,
                 image_set: str,
                 download: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super().__init__(root, year, image_set, download, None, None, None)
        self._transform = transform
        self._target_transform = target_transform
        self._transforms = transforms
        self.voc_box = lambda x: [int(x['xmin']), int(x['ymin']), int(x['xmax']), int(x['ymax'])]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, ann = super().__getitem__(idx)
        W, H = img.size
        if self._transform or self._target_transform or self._transforms:
            objects = ann['annotation']['object']
            bboxes = []
            names = []
            for obj in objects:
                names.append(VOC_N2I[obj['name']])
                bboxes.append(self.voc_box(obj['bndbox']))
            bboxes = torch.Tensor(bboxes)
            bboxes = tv_tensors.BoundingBoxes(bboxes, format = 'XYXY', canvas_size = (H, W))
            if self._transform and self._target_transform:
                img = self._transofrm(img)
                bboxes = self._target_transform(bboxes)
            elif self._transforms:
                img, bboxes = self._transforms(img, bboxes)
            else:
                raise Exception('We need transform for both image and targets')
            return (img, {'filename': ann['annotation']['filename'], 'image_size': (H, W),'boxes': bboxes, 'labels': torch.tensor(names)})
        else:
            return img, ann

if __name__ == '__main__':
    dataset = VOCDetection()