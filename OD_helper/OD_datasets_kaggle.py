from typing import Callable, Any, Tuple, Optional, Dict, List

import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2

import numpy as np

import os
import shutil
import collections
import PIL
from PIL import Image

from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

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

from typing import Callable, Any, Tuple, Optional, Dict, List
import collections

class KaggleVOCDetection(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 image_set: str = 'train',
                 transforms: Optional[Callable] = None):
        super().__init__()
        self.transforms = transforms
        voc_root = root
        splits_dir = os.path.join(voc_root, "ImageSets", "Main")
        split_f = os.path.join(splits_dir, image_set.rstrip("\n")+'.txt')
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]

        assert len(self.images) == len(self.targets)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(KaggleVOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    @property
    def annotations(self) -> List[str]:
        return self.targets

class VOCDetection(KaggleVOCDetection):
    def __init__(self,
                 root: str,
                 image_set: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super().__init__(root, image_set, None)
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