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
from utils.cython_bbox import bbox_overlaps
def main():
  ssd_net = build_face_ssd('val')
  net = ssd_net
  net.load_state_dict(torch.load('./checkpoints/SSD_wFace_127.pth'))
  net.eval()

  dataset = WiderFaceDetection(root='/data00/kangyang/datasets/WIDER_FACE', transform=SimpleTestAugmentation())
  data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False, collate_fn=detection_collate, pin_memory=True)
  cpu = False

  fp = 0; tp = 0; fn = 0
  if not cpu:
    net = net.cuda()
  for i_batch, (images, targets) in enumerate(data_loader):
    print i_batch

    if cpu:
      pass
    else:
      with torch.no_grad():
        blob = Variable(images.cuda())

    y = net(blob)
    face_detections = y.data[0][1]
    num_boxes = face_detections.shape[0]
    count = 0
    im = np.transpose(images[0], (1,2,0)).numpy().astype(np.uint8)
    im = im[:,:,::-1]
    im = np.ascontiguousarray(im)
    im_height, im_width, _ = im.shape

    pred_boxes = []
    for b in range(num_boxes):
      if face_detections[b,0] >= 0.3:
        box = face_detections[b,1:].numpy()
        x1 = min(im_width, max(0, box[0] * im_width));
        y1 = min(im_height, max(0, box[1] * im_height));
        x2 = min(im_width, max(0, box[2] * im_width));
        y2 = min(im_height, max(0, box[3] * im_height));
        im = cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0))
        pred_boxes.append([x1,y1,x2,y2])
        count += 1

    res_path = './res/test_{}.png'.format(i_batch)
    pred_boxes = np.array(pred_boxes)
    gt = targets[0].numpy()
    num_gt = gt.shape[0]

    if num_gt == 0:
      fp += num_pred
      print 'tp: {}, fp: {}, fn: {}'.format(tp, fp, fn)
      continue

    gt[:, 0] = np.minimum(im_width, np.maximum(0, gt[:, 0] * im_width))
    gt[:, 1] = np.minimum(im_height, np.maximum(0, gt[:, 1] * im_height))
    gt[:, 2] = np.minimum(im_width, np.maximum(0, gt[:, 2] * im_width))
    gt[:, 3] = np.minimum(im_height, np.maximum(0, gt[:, 3] * im_height))
    for n_gt in range(gt.shape[0]):
      gt_box = gt[n_gt, :]
      im = cv2.rectangle(im, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0,0,255))
    cv2.imwrite(res_path, im)
    print res_path

    num_gt = gt.shape[0]
    num_pred = pred_boxes.shape[0]
    if num_pred == 0:
      fn += num_gt
      print 'tp: {}, fp: {}, fn: {}'.format(tp, fp, fn)
      continue

    gt_assign_preds = []
    for _ in range(num_gt):
      gt_assign_preds.append([])
    overlaps = bbox_overlaps(pred_boxes.astype(np.float64), gt.astype(np.float64))
    argmax_gts = overlaps.argmax(axis=1)

    for p in range(num_pred):
      argmax_gt = argmax_gts[p]
      gt_ov_pred = overlaps[p, argmax_gt]
      if gt_ov_pred >= 0.5:
        gt_assign_preds[argmax_gt].append(p)
      else:
        fp += 1

    for g in range(num_gt):
      gt_assign_pred = gt_assign_preds[g]
      if len(gt_assign_pred) == 0:
        fn += 1
      elif len(gt_assign_pred) == 1:
        tp += 1
      elif len(gt_assign_pred) > 1:
        tp += 1
        fp += (len(gt_assign_pred) - 1)
      else:
        assert(False)

    print 'get fgboxes: {}'.format(count)
    print 'tp: {}, fp: {}, fn: {}'.format(tp, fp, fn)

  precision = tp * 1.0 / (tp + fp)
  recall = tp * 1.0 / (tp + fn)

  print 'precision: {}, recall: {}'.format(precision, recall)


if __name__ == '__main__':
  main()
