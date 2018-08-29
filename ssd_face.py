import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import widerface, config
import os

class SSD(nn.Module):
  def load_weights(self, base_file):
      other, ext = os.path.splitext(base_file)
      if ext == '.pkl' or '.pth':
        print('Loading weights into state dict...')
        self.load_state_dict(torch.load(base_file,
          map_location=lambda storage, loc: storage))
        print('Finished!')
      else:
        print('Sorry only .pth and .pkl files supported.')

  def __init__(self, phase, base, extras, head, num_classes):
    super(SSD, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.cfg = widerface
    self.priorbox = PriorFaceBox(self.cfg)
    with torch.no_grad():
      self.priors = Variable(self.priorbox.forward().cuda())

    self.vgg = nn.ModuleList(base)
    self.L2Norm = L2Norm(512, 20)
    self.extras = nn.ModuleList(extras)

    self.loc = nn.ModuleList(head[0])
    self.conf = nn.ModuleList(head[1])

    if phase != 'train':
      self.softmax = nn.Softmax(dim=-1)
      self.detect = Detect(2, 0, 200, 0.01, 0.3)

  def forward(self, x):
    sources = list()
    loc = list()
    conf = list()

    for k in range(23):
      x = self.vgg[k](x)

    s = self.L2Norm(x)
    sources.append(s)

    for k in range(23, len(self.vgg)):
      x = self.vgg[k](x)
    sources.append(x)

    for k, v in enumerate(self.extras):
      x = F.relu(v(x), inplace=True)
      if k % 2 == 1:
        sources.append(x)

    for (x, l, c) in zip(sources, self.loc, self.conf):
      loc.append(l(x).permute(0, 2, 3, 1).contiguous())
      conf.append(c(x).permute(0, 2, 3, 1).contiguous())

    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase != 'train':
      output = self.detect(loc.view(loc.size(0), -1, 4),
          self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
          self.priors)
    else:
      output = (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors)
    return output

def vgg(cfg, i, batch_norm=False):
  layers = []
  in_channels = i
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    elif v == 'C':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
  conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
  conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
  layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
  return layers

def add_extras(cfg, i, batch_norm=False):
  layers = []
  in_channels = i
  flag = False
  for k, v in enumerate(cfg):
    if in_channels != 'S':
      if v == 'S':
        layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
      else:
        layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
      flag = not flag
    in_channels = v
  return layers

def multibox(vgg, extra_layers, cfg, num_classes):
  loc_layers = []
  conf_layers = []
  vgg_source = [21, -2]

  for k, v in enumerate(vgg_source):
    loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
  for k, v in enumerate(extra_layers[1::2], 2):
    loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
  return vgg, extra_layers, (loc_layers, conf_layers)

base = {
  'VGG_16_reduced': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
}
extras = {
  'VGG_16_reduced': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
  'VGG_16_reduced': [2,4,4,4,4],
}

def build_face_ssd(phase):
  base_, extras_, head_ = multibox(vgg(base['VGG_16_reduced'], 3, False),
      add_extras(extras['VGG_16_reduced'], 1024),
      mbox['VGG_16_reduced'], 2)

  ssd_net = SSD(phase, base_, extras_, head_, 2)
  return ssd_net

if __name__ == '__main__':
  build_face_ssd('train')
