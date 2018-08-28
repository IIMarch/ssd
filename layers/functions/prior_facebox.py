from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorFaceBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorFaceBox, self).__init__()
        self.image_size = cfg['input_shape']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance']
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1]), repeat=1):
                f_hk = self.image_size[0] * 1.0 / self.steps[k]
                f_wk = self.image_size[1] * 1.0 / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_wk
                cy = (i + 0.5) / f_hk

                # aspect_ratio: 1
                # rel size: min_size
                s_hk = self.min_sizes[k] * 1.0 / self.image_size[0]
                s_wk = self.min_sizes[k] * 1.0 / self.image_size[1]
                mean += [cx, cy, s_wk, s_hk]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_hk_prime = sqrt(s_hk * (self.max_sizes[k] / self.image_size[0]))
                s_wk_prime = sqrt(s_wk * (self.max_sizes[k] / self.image_size[1]))
                mean += [cx, cy, s_wk_prime, s_hk_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    if ar == 1:
                      continue
                    mean += [cx, cy, s_wk*sqrt(ar), s_hk/sqrt(ar)]
                    mean += [cx, cy, s_wk/sqrt(ar), s_hk*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
