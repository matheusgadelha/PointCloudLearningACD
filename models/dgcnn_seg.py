#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Aruni RC
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class get_model(nn.Module):
    def __init__(self, num_classes, k=20, emb_dims=1024, dropout=0.5, 
                 normal_channel=False, l2_norm=False):
        super(get_model, self).__init__()

    # def __init__(self, args, output_channels=40):
    #     super(DGCNN, self).__init__()
        # self.args = args
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(emb_dims)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)
        # edge-conv layers
        self.edge1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, ),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.edge2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.edge3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        # global 1x1 conv
        self.conv1 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        # per-point 1x1 convs
        self.conv2 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.cls1 = nn.Conv1d(128, num_classes, kernel_size=1)
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)


    def forward(self, x, cls_label=[], keep_unpooled=False):
        l1_pts = [] # dummy
        feat = [] # dummy
        batch_size = x.size(0)
        num_pts = x.size(-1)
        x = get_graph_feature(x, k=self.k)
        x = self.edge1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # [bs, 64, 2048]
        if keep_unpooled:
            out1 = x

        x = get_graph_feature(x1, k=self.k)
        x = self.edge2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # [bs, 64, 2048]
        if keep_unpooled:
            out2 = x

        x = get_graph_feature(x2, k=self.k)
        x = self.edge3(x)
        x3 = x.max(dim=-1, keepdim=False)[0] # [bs, 64, 2048]
        if keep_unpooled:
            out3 = x

        x = torch.cat((x1, x2, x3), dim=1) # [bs, 192, 2048]
        x = self.conv1(x)  # [bs, 1024, 2048]
             
        out_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # [bs, 1024]
        out_expand = out_max.unsqueeze(-1).repeat([1, 1, num_pts])  # [bs, 1024, 2048]        
        x = torch.cat((out_expand, x1, x2, x3), dim=1)  # [bs, 1216, 2048]
        x = self.conv2(x)  # [bs, 256, 2048]
        x = self.dp1(x)
        x = self.conv3(x)  # [bs, 256, 2048]
        x = self.dp2(x)
        feat = self.conv4(x)  # [bs, 128, 2048]

        # classification (per point)
        x = self.cls1(feat)
        x = F.log_softmax(x, dim=1)        
        x = x.permute(0, 2, 1)

        if keep_unpooled:
            ret = x, (out1, out2, out3), feat
        else:
            ret = x, (x1, x2, x3), feat
        return ret



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


class get_selfsup_loss(nn.Module):
    def __init__(self, margin=0.5):
        super(get_selfsup_loss, self).__init__()
        self.margin = margin

    def forward(self, feat, target):
        feat = F.normalize(feat, p=2, dim=1)
        pair_sim = torch.bmm(feat.transpose(1,2), feat)

        one_hot_target = F.one_hot(target).float()
        pair_target = torch.bmm(one_hot_target, one_hot_target.transpose(1,2))        

        cosine_loss = pair_target * (1. - pair_sim) + (1. - pair_target) * F.relu(pair_sim - self.margin)
        diag_mask = 1 - torch.eye(cosine_loss.shape[-1])  # discard diag elems (always 1)

        with torch.no_grad():
            # balance positive and negative pairs
            pos_fraction = (pair_target.data == 1).float().mean()
            sample_neg = torch.cuda.FloatTensor(*pair_target.shape).uniform_() > 1 - pos_fraction
            sample_mask = (pair_target.data == 1) | sample_neg # all positives, sampled negatives

        cosine_loss = diag_mask.unsqueeze(0).cuda() * sample_mask * cosine_loss 
        total_loss = 0.5 * cosine_loss.mean() # scale down

        return total_loss