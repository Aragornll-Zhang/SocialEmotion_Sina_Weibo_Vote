# -*- coding: UTF-8 -*-
import sys
PATH1 = '/root/MLabelCls'
if PATH1 not  in sys.path:
    sys.path.insert(0,PATH1)
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
from sklearn import metrics

from GCN.Graph_init import *
from GCN.GCN_dataLoader import *
import math
from GCN.util import *


# GCN -> |labels|个分类器  +  简单内积法
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Features:(label embedding) input: X \in R^{C*C} , Weight \in R^{C* out_feature} , A:re-weighted adjacent matrix
        #  output: A * X * W(\theta)  \in R^{C * out_feature}
        support = torch.mm( input, self.weight)
        output = torch.mm( adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self , in_feature ,hidden_size, out_feature = 1024, layer_num = 2 , dropout = 0.2 ):
        super(GCN,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # get GCN layers:
        self.layer_num = layer_num

        self.gc1 = GraphConvolution(in_feature, out_feature) if layer_num == 1 else GraphConvolution(in_feature , hidden_size)
        self.gc2 = GraphConvolution(hidden_size, out_feature) if layer_num == 2 else GraphConvolution(hidden_size,  hidden_size)
        self.gc3 = GraphConvolution(hidden_size, out_feature) if layer_num == 3 else GraphConvolution(hidden_size , hidden_size)
        self.gc4 = GraphConvolution(hidden_size, out_feature) if layer_num == 4 else GraphConvolution(hidden_size , hidden_size)
        self.gc5 = GraphConvolution(hidden_size, out_feature)
        return

    def forward(self, x , adj):
        h = self.gc1(x, adj)
        if self.layer_num >= 2:
            h = self.dropout(F.relu(h))
            h= self.gc2(h , adj)
        if self.layer_num >= 3:
            h = self.dropout(F.relu(h))
            h= self.gc3(h , adj)
        if self.layer_num >= 4:
            h = self.dropout(F.relu(h))
            h= self.gc4(h , adj)
        if self.layer_num >= 5:
            h = self.dropout(F.relu(h))
            h= self.gc5(h , adj)
        return h


class Bert_GCN(nn.Module):
    def __init__(self , bert_model , opt = Bert_Config(), label_embedding = None , A = None ):
        super(Bert_GCN,self).__init__()
        self.bert = bert_model
        self.If_GCN = opt.If_GCN
        if self.If_GCN:
            self.gcn = GCN(   in_feature =  opt.GCN_emb_size,
                              hidden_size = opt.GCN_hidden_size ,
                              out_feature = opt.hidden_dim ,
                              layer_num=opt.GCN_layer)
        else:
            self.linear = nn.Linear(opt.hidden_dim , opt.label_num , bias=False)
            # # 消融实验
            # self.linear = nn.Sequential( nn.Linear(opt.hidden_dim , opt.out_feature) ,
            #                              nn.ReLU(),
            #                              nn.Linear(opt.out_feature , opt.label_num , bias=False))

        if label_embedding is None:
            self.Node_attr = nn.Parameter( torch.randn(opt.label_num , opt.GCN_emb_size)  , requires_grad=True   )
        else:
            self.Node_attr = nn.Parameter( label_embedding , requires_grad=True )
        if A is not None:
            self.A = nn.Parameter( A , requires_grad=False )

        self.stop_bert_train = False

    def forward(self ,x, mask):
        output = self.bert(x , mask)[1]
        if self.stop_bert_train:
            output = output.detach() # 先不训 bert 部分, 等等GCN老伙计
        if self.If_GCN:
            Linear = self.gcn( self.Node_attr , self.A).transpose(0,1)
            output =  torch.matmul(output , Linear)
            return output # torch.sigmoid( output )
        else:
            return self.linear(output) # torch.sigmoid( self.linear(output) )

    def chg_BackPropagation_state(self , new_state: bool):
        self.stop_bert_train = new_state
        return
