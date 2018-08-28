from data import *
from utils.augmentations import SSDAugmentation, SimpleAugmentation
from layers.modules import MultiBoxLoss
from ssd_face import build_face_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import cv2
from visdom import Visdom

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='WIDER_FACE', choices=['VOC', 'COCO', 'WIDER_FACE'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/data00/kangyang/datasets/WIDER_FACE',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.000125, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    viz = Visdom(port=8097)
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        #if args.dataset_root == COCO_ROOT:
        #    parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'WIDER_FACE':
        dataset = WiderFaceDetection(root=args.dataset_root, transform=SimpleAugmentation())

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    ssd_net = build_face_ssd('train')
    if args.cuda:
      net = torch.nn.DataParallel(ssd_net)
      cudnn.benchmark = True
    else:
      net = ssd_net

    if args.resume:
      print('Resuming training, loading {}...'.format(args.resume))
      ssd_net.load_weights(args.resume)
    else:
      vgg_weights = torch.load(args.save_folder + args.basenet)
      print('Loading base network...')
      ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
      net = net.cuda()

    if not args.resume:
      print('Initializing weights...')
      ssd_net.extras.apply(weights_init)
      ssd_net.loc.apply(weights_init)
      ssd_net.conf.apply(weights_init)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.00005)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes=2, overlap_thresh=0.5, prior_for_matching=True,
        bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False,
        use_gpu=args.cuda)

    net.train()

    loc_loss = 0
    conf_loss = 0
    epoch = 0
    epoch_size = len(dataset) // args.batch_size
    step_index = 0

    height = 512
    width = 512
    in_pipe = viz.images(np.random.randn(args.batch_size, 3, height, width))
    # create batch iterator
    lr = args.lr
    acum_loss = 0.0
    count = 0
    iteration = args.start_iter
    for epoch in range(100000):
      for i_batch, (images, targets) in enumerate(data_loader):
        if i_batch % 20 == 0:
          #height, width = images.shape[2], images.shape[3]
          np_im = images.numpy().copy() + 128
          #drawes = []
          #for bb in range(args.batch_size):
          #  im = np.ascontiguousarray(np.transpose(np_im[bb], (1,2,0)))
          ##  boxes = targets[bb].numpy().copy()
          #  num_boxes = boxes.shape[0]
          #  for kk in range(num_boxes):
          #    x1 = int(min(width, max(0, width * boxes[kk,0])))
          #    y1 = int(min(height, max(0, height * boxes[kk,1])))
          #    x2 = int(min(width, max(0, width * boxes[kk,2])))
          #    y2 = int(min(height, max(0, height * boxes[kk,3])))
          #    cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)
          #  drawes.append(im)
          #drawes = np.transpose(np.stack(drawes, axis=0), (0,3,1,2))
          #viz.images(drawes, win=in_pipe)


        if args.cuda:
          with torch.no_grad():
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
        else:
          with torch.no_grad():
            images = Variable(images)
            targets = [Variable(ann) for ann in targets]
        out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c, need_draw = criterion(out, targets, images.shape, i_batch)
        loss = loss_l + loss_c
        acum_loss += loss.data.item()
        count += 1
        loss.backward()
        optimizer.step()

        if i_batch % 20 == 0:
        #if False:
          height, width = images.shape[2], images.shape[3]
          drawes = []
          for bb in range(args.batch_size):
            im = np.ascontiguousarray(np.transpose(np_im[bb], (1,2,0)))
            boxes = targets[bb].cpu().numpy()
            num_boxes = boxes.shape[0]
            matches = need_draw[bb]
            for kk in range(num_boxes):
              x1 = int(min(width, max(0, width * boxes[kk,0])))
              y1 = int(min(height, max(0, height * boxes[kk,1])))
              x2 = int(min(width, max(0, width * boxes[kk,2])))
              y2 = int(min(height, max(0, height * boxes[kk,3])))
              cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)

            for kk in range(len(matches)):
              x1,y1,x2,y2 = matches[kk]
              cv2.rectangle(im, (x1, y1), (x2, y2), (0,0,255), 2)
            drawes.append(im)
          drawes = np.transpose(np.stack(drawes, axis=0), (0,3,1,2))
          viz.images(drawes, win=in_pipe)

        if i_batch % 5 == 0:
          print 'batch: %d, loss_loc: %.4f, loss_cls: %.4f, acum_loss: %4f' % (i_batch, loss_l.data.item(),
              loss_c.data.item(), acum_loss * 1.0 / count)
          for param_group in optimizer.param_groups:
            print 'lr: {}'.format(param_group['lr'])

      if epoch in [900, 1800]:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
      print ('saving SSD_wFace_matching_moreprior_{}.pth'.format(epoch))
      torch.save(ssd_net.state_dict(), './checkpoints/SSD_wFace_matching_moreprior_{}.pth'.format(epoch))

        #if i_batch > 100:
        #  break
        #images = images.numpy()[0]
        #images = np.ascontiguousarray(np.transpose(images,(1,2,0)).astype(np.uint8)[:,:,::-1])
        #num_batch = len(targets)
        #for b in range(num_batch):
        #  boxes = targets[b].numpy()
        #  num_boxes = boxes.shape[0]
        #  for k in range(num_boxes):
        #    box = boxes[k,:]
        #    cv2.rectangle(images, (int(box[0]), int(box[1])),
        #        (int(box[2]), int(box[3])), (0,0,255));

        #  cv2.imwrite("{}.png".format(i_batch), images)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        m.bias.data.zero_()

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
