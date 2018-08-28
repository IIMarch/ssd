from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
  def __init__(self, root, image_sets=['train'], transform=None, target_transform=None, dataset_name='WIDER_FACE'):
    self.root = root
    self.image_set = image_sets
    self.transform = transform
    self.target_transform = target_transform
    self.name = dataset_name
    self._annopath = osp.join(self.root, 'wider_face_split/wider_face_{}_bbx_gt.txt'.format(image_sets[0]))
    self._load_anno()
    self.image_base = osp.join(self.root, 'WIDER_{}/images'.format(image_sets[0]))

  def __getitem__(self, index):
    im, gt, h, w = self.pull_item(index)

    return im, gt

  def __len__(self):
    return len(self.ids)

  def pull_item(self, index):
    img_id = self.ids[index]
    image_path = osp.join(self.image_base, img_id)

    img = cv2.imread(image_path)

    height, width, channels = img.shape
    target = self.targets[index]

    if self.target_transform is not None:
      target = self.target_transform(target, width, height)

    if self.transform is not None:
      img, boxes, labels = self.transform(img, target[:, :4], target[:, 4:5])

      img = img[:, :, (2, 1, 0)]
      target = np.hstack((boxes, labels))

    return torch.from_numpy(img).permute(2, 0, 1), target, height, width

  def _load_anno(self):
    self.ids = []
    self.targets = []
    with open(self._annopath, 'r') as fid:
      line = fid.readline()
      while line != '':
        image_p = line.strip()
        self.ids.append(image_p)

        num_boxes = int(fid.readline().strip())
        valid_boxes = []
        for i in range(num_boxes):
          splited = fid.readline().strip().split(' ')
          x1 = int(splited[0]); y1 = int(splited[1]);
          x2 = int(splited[2]) + x1 - 1; y2 = int(splited[3]) + y1 - 1;
          blur = int(splited[4]); expression = int(splited[5])
          occlusion = int(splited[6]); pose = int(splited[7])
          invalid = int(splited[8])

          #if occlusion != 0:
          #  continue
          valid_boxes.append([x1, y1, x2, y2, 1, blur, expression, occlusion, pose, invalid])
        line = fid.readline()
        if len(valid_boxes) == 0:
          self.targets.append(np.zeros([0, 10], dtype=np.float32))
        else:
          self.targets.append(np.array(valid_boxes).astype(np.float32))







