import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils.augmentations import SimpleAugmentation, SimpleTestAugmentation
from ssd_face import build_face_ssd
from data import *
import torch.utils.data as data

def main():
  ssd_net = build_face_ssd('val')
  net = ssd_net
  net.load_state_dict(torch.load('./checkpoints/SSD_wFace_matching_0.6_128.pth'))
  net.eval()

  dataset = WiderFaceDetection(root='/data00/kangyang/datasets/WIDER_FACE', transform=SimpleTestAugmentation())
  data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False, collate_fn=detection_collate, pin_memory=True)
  cpu = False

  if not cpu:
    net = net.cuda()
  for i_batch, (images, targets) in enumerate(data_loader):

    if cpu:
      pass
    else:
      with torch.no_grad():
        blob = Variable(images.cuda())
        targets = [Variable(ann) for ann in targets]

    y = net(blob)
    face_detections = y.data[0][1]
    num_boxes = face_detections.shape[0]
    count = 0
    im = np.transpose(images[0], (1,2,0)).numpy().astype(np.uint8) + 128
    im = im[:,:,::-1]
    im = np.ascontiguousarray(im)
    im_height, im_width, _ = im.shape

    for b in range(num_boxes):
      if face_detections[b,0] >= 0.3:
        box = face_detections[b,1:]
        x1 = min(im_width, max(0, box[0] * im_width));
        y1 = min(im_height, max(0, box[1] * im_height));
        x2 = min(im_width, max(0, box[2] * im_width));
        y2 = min(im_height, max(0, box[3] * im_height));
        im = cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0))
        count += 1

    print 'get fgboxes: {}'.format(count)

    cv2.imwrite('test_{}.png'.format(i_batch), im)
    if i_batch > 100:
      break


if __name__ == '__main__':
  main()
