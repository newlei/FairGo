# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys 

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
from collections import defaultdict
import time 
from shutil import copyfile
import pickle

class DisClfGender(nn.Module):
    def __init__(self,embed_dim,out_dim,attribute,use_cross_entropy=True):
        super(DisClfGender, self).__init__()
        self.embed_dim = int(embed_dim) 
        self.attribute = attribute
        self.criterion = nn.CrossEntropyLoss()#torch.nn.BCELoss()#nn.CrossEntropyLoss()   
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/4), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/4), int(self.embed_dim/8), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Sigmoid(),
            nn.Linear(int(self.embed_dim /8), self.out_dim , bias=True),
            # nn.Sigmoid(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True)
            # nn.Sigmoid()
            )
    def forward(self, ents_emb, labels, return_loss=True):
        scores = self.net(ents_emb)
        outputs = F.log_softmax(scores, dim=1)
        # pdb.set_trace()
        if return_loss:
            loss = self.criterion(outputs, labels)
            # pdb.set_trace()
            return loss
        else:
            return outputs,labels


class DisClfAge(nn.Module):
    def __init__(self,embed_dim,out_dim,attribute,use_cross_entropy=True):
        super(DisClfAge, self).__init__()
        self.embed_dim = int(embed_dim) 
        self.attribute = attribute
        self.criterion = nn.CrossEntropyLoss()#torch.nn.BCELoss()#nn.CrossEntropyLoss()  
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/2), int(self.embed_dim/4), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Sigmoid(),
            nn.Linear(int(self.embed_dim /4), self.out_dim , bias=True),
            # nn.Sigmoid(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True)
            # nn.Sigmoid()
            )
    def forward(self, ents_emb, labels, return_loss=True):
        scores = self.net(ents_emb)
        outputs = F.log_softmax(scores, dim=1)
        # pdb.set_trace()
        if return_loss:
            loss = self.criterion(outputs, labels)
            # pdb.set_trace()
            return loss
        else:
            return outputs,labels




class DisClfOcc(nn.Module):
    def __init__(self,embed_dim,out_dim,attribute,use_cross_entropy=True):
        super(DisClfOcc, self).__init__()
        self.embed_dim = int(embed_dim) 
        self.attribute = attribute
        self.criterion = nn.CrossEntropyLoss()#torch.nn.BCELoss()#nn.CrossEntropyLoss()  
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/2), self.out_dim, bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Sigmoid(),
            # nn.Linear(int(self.embed_dim /4), self.out_dim , bias=True),
            # nn.Sigmoid(),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True)
            # nn.Sigmoid()
            )

    def forward(self, ents_emb, labels, return_loss=True):
        scores = self.net(ents_emb)
        outputs = F.log_softmax(scores, dim=1)
        # pdb.set_trace()
        if return_loss:
            loss = self.criterion(outputs, labels)
            # pdb.set_trace()
            return loss
        else:
            return outputs,labels



class AttributeFilter(nn.Module):
    def __init__(self, embed_dim, attribute='gender'):
        super(AttributeFilter, self).__init__()
        self.embed_dim = embed_dim
        self.attribute = attribute 
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim*2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), self.embed_dim, bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Sigmoid(),
            nn.Linear(self.embed_dim, self.embed_dim , bias=True),
            # nn.Sigmoid()
            )
        self.fc1 = nn.Linear(self.embed_dim, int(self.embed_dim/4), bias=True)
        self.fc2 = nn.Linear(int(self.embed_dim /4), self.embed_dim, bias=True)
        self.fc3 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # self.batchnorm = nn.BatchNorm1d(self.embed_dim,track_running_stats=True)

    def forward(self, ents_emb):
        # h0 = F.leaky_relu(self.W0(ents_emb))
        # h1 = F.leaky_relu(self.fc1(ents_emb))#self.fc1(ents_emb)#
        # h2 = F.leaky_relu(self.fc2(h1))
        # h3 = (self.fc3(h2))
        # h2 = self.batchnorm(h2)
        h3 = self.net(ents_emb)
        return h3

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class LineGCN(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(LineGCN, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_num=user_num
        self.item_num=item_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        self.factor_num=factor_num  

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  
  
    def forward(self,user_item_matrix,item_user_matrix,d_i_train,d_j_train): 
        
        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        # pdb.set_trace()
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]] 
            
        d_i_train0=torch.cuda.FloatTensor(d_i_train)
        d_j_train0=torch.cuda.FloatTensor(d_j_train)
        # print(d_i_train0.shape,d_j_train0.shape)
        # pdb.set_trace() 
        d_i_train1=d_i_train0.expand(-1,self.factor_num)
        d_j_train1=d_j_train0.expand(-1,self.factor_num)

        users_embedding=self.embed_user.weight#torch.cat((self.embed_user.weight, users_features0),1)
        items_embedding=self.embed_item.weight#torch.cat((self.embed_item.weight, items_features0),1) 

        gcn1_users_embedding = (torch.sparse.mm(user_item_matrix, items_embedding) + users_embedding.mul(d_i_train1))#*2. #+ users_embedding
        gcn1_items_embedding = (torch.sparse.mm(item_user_matrix, users_embedding) + items_embedding.mul(d_j_train1))#*2. #+ items_embedding

        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding),-1)#+gcn4_items_embedding#
        
 
        return gcn_users_embedding,gcn_items_embedding#,torch.unsqueeze(torch.cat((gcn_users_embedding, gcn_items_embedding),0), 0)




 