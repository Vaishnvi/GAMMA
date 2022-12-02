# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

from torch.nn.utils import spectral_norm

from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
        #print(m,"\n")

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


#def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    #return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   #stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

class self_attn2(nn.Module):
    #Self attention Layer"""

    def __init__(self,in_channels):
        super(self_attn2, self).__init__()
        self.in_channels = in_channels

        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0).cuda()
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0).cuda()
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0).cuda()
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0).cuda()
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0).cuda()
        self.softmax  = nn.Softmax(dim=-1).cuda()
        self.sigma = nn.Parameter(torch.zeros(1)).cuda()
        self._init_weights()

        #print(self_attn2.snconv1x1_attn.weight_orig)

    def _init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
            #print(m,"\n")

    def forward(self, x):
        x = x.cuda()
        #with torch.no_grad():

        self.apply(init_weights)

        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        #print(theta.dtype)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        #print("\nx:",x.shape)
        phi = self.snconv1x1_phi(x)
        #print("phiconv:",phi.shape)
        phi = self.maxpool(phi)
        #print("phimax:",phi.shape)
        #phi = phi.view(-1, ch//8, h*w//4)
        phi = phi.view(-1, ch//8, phi.shape[2]*phi.shape[3])
        #print("phiview:",phi.shape)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        #g = g.view(-1, ch//2, h*w//4)
        g = g.view(-1, ch//2, g.shape[2]*g.shape[3])
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        #attn_g = attn_g.cuda()
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)

        #print("\n________________WRITING_____________\n")
        #print("\n____Done____\n")


        #Out
        out = x + self.sigma*attn_g
        out = out.cuda()
        #out1 = Variable(out, requires_grad = True).cuda(cuda0)
        #out1 = out.cuda(cuda0)

        del self.snconv1x1_theta,
        self.snconv1x1_phi,
        self.snconv1x1_g,
        self.snconv1x1_attn,
        self.maxpool,   self.softmax ,
        self.sigma

        del x, theta, phi, attn, g
        del self
        ##del cud
        torch.cuda.empty_cache()
        
        return out


class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    print("\nVGG",vgg.classifier)
    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
    self.self_attn2 = self_attn2(512) 
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

