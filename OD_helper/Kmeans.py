from typing import Any, Optional

import torch
import torchvision

from tqdm import tqdm

class KMeans:
    def __init__(self,
                 n_clusters: int,
                 distance: Optional[Any] = None):
      self.n_clusters = n_clusters
      if distance is None or distance == 'euclid':
          self.cdist = self._distance
      else:
          self.cdist = distance

    def __call__(self,
                 X: torch.Tensor,
                 num_iters: int):
        centroids = self._init_centroids(X)
        for i in tqdm(range(num_iters)):
            distance_mat = self.cdist(X, centroids)
            Y = self._update_labels(distance_mat)
            centroids = self._update_centroids(X, Y)
        return centroids, Y

    def _distance(self,
                  X: torch.Tensor,
                  centroids: torch.Tensor):
        """
        samples: (N, d)
        centroid: (m, d)
        """
        return (torch.sqrt(((X.unsqueeze(0) - centroids.unsqueeze(1))**2).sum(dim = -1))).transpose(0, 1)

    def _init_centroids(self,
                        X: torch.Tensor):
        centroids = X[torch.randint(0, X.shape[0], (self.n_clusters,))]
        return centroids

    def _update_labels(self,
                       distance_mat: torch.Tensor):
        return torch.argmin(distance_mat, dim = -1)

    def _update_centroids(self,
                          X: torch.Tensor,
                          Y: torch.Tensor):
        centroids = []
        for i in range(self.n_clusters):
            centroids.append(torch.mean(X[Y == i], dim = 0))
        return torch.stack(centroids)

def iou_distance(boxes: torch.Tensor,
                 centroids: torch.Tensor):
    """
    boxes: (N, 4)
    centroids: (m, 4)
    """
    ious = torchvision.ops.box_iou(boxes, centroids)
    return 1 - ious