# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys


# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [1]))

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd 
import torch.utils.data as data

from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import pdb
import copy 
from collections import defaultdict
import time
import data_utils 
from shutil import copyfile
import pickle 
import filter_layer


class Classifier(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(Classifier, self).__init__()
        """ 
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator
        """     
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/4), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Linear(int(self.embed_dim/4), int(self.embed_dim/8), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Linear(int(self.embed_dim /8), self.out_dim , bias=True),
            # nn.Sigmoid(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True),
            nn.Sigmoid()
            )
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
    def forward(self, emb0,label0):
        scores = self.net(emb0)
        # outputs = F.log_softmax(scores, dim=1)
        label0 =label0.view(-1)
        loss = self.criterion(scores, label0)
        # pdb.set_trace()
        return loss 

    def prediction(self, emb0): 
        scores = self.net(emb0) 
        return scores.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class ClassifyData(data.Dataset):
    def __init__(self,data_set_count=0, train_data=None, is_training=None,embed_dim=0):
        super(ClassifyData, self).__init__() 
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.features_fill = train_data 
        self.embed_dim = embed_dim
    def __len__(self):  
        return self.data_set_count#return self.num_ng*len(self.train_dict)
          
    def __getitem__(self, idx):
        features = self.features_fill
        feature_user = features[idx][:self.embed_dim ]
        label_user = features[idx][self.embed_dim:]
        label_user = label_user.astype(np.float32)
        return feature_user, label_user


def clf_gender_all_pre(model_id,epoch_run,users_embs,factor_num):
    ##movieLens-1M
    # user_num=6040#user_size
    # item_num=3952#item_size 
    # factor_num=64
    batch_size=2048*100  
    dataset_base_path='../../data/ml1m'

    epoch_id='clf_gender/'+str(epoch_run) 
    print(model_id,epoch_id)
    dataset='movieLens-1M'
 
    users_features = np.load(dataset_base_path+'/data1t5/users_features_list.npy',allow_pickle=True)

    users_features=users_features.astype(np.float32)
    users_features_age_oh=users_features[:,:2]
    users_features_age=[np.where(r==1)[0][0] for r in users_features_age_oh]
    users_features_age=np.array(users_features_age).astype(np.float32)
    users_features_age = (users_features_age).reshape(-1,1)

    users_embs_cat_att = np.concatenate((users_embs, users_features_age), axis=-1)
    np.random.shuffle(users_embs_cat_att)
    #6014=
    train_data_all = users_embs_cat_att[:-1000][:]
    training_count=len(train_data_all)
    test_data_all = users_embs_cat_att[-1000:][:]
    testing_count=len(test_data_all)
     
    train_dataset = ClassifyData(
            data_set_count=training_count, train_data=train_data_all,is_training = True,embed_dim=factor_num)
    train_loader = DataLoader(train_dataset,
            batch_size=training_count, shuffle=True, num_workers=2)

    testing_dataset_loss = ClassifyData(
            data_set_count=testing_count, train_data=test_data_all,is_training = False,embed_dim=factor_num)
    testing_loader = DataLoader(testing_dataset_loss,
            batch_size=testing_count, shuffle=False, num_workers=0)


    ######################################################## TRAINING #####################################
    # print('--------training processing-------')
    count, best_hr = 0, 0 
    model = Classifier(embed_dim=factor_num, out_dim=1)
    model=model.to('cuda')  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, betas=(0.5, 0.99))  
 
    current_loss=0
    flag_stop=-1
    res_center=''
    res_auc=[]
    for epoch in range(230):
        model.train()  
        start_time = time.time() 
        # print('train data of ng_sample is  end')
        train_loss_sum=[]
        train_loss_bpr=[]  
        for user_features, user_labels in train_loader:
            # pdb.set_trace()
            user_features = user_features.cuda()
            user_labels = user_labels.cuda() 
            
            loss_get = model(user_features,user_labels)  
            optimizer.zero_grad()
            loss_get.backward()
            optimizer.step()  
            count += 1
            train_loss_sum.append(loss_get.item()) 

        elapsed_time = time.time() - start_time
        train_loss=round(np.mean(train_loss_sum),4)# 
        str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+' train loss:'+str(train_loss) 
        if train_loss>current_loss:
            flag_stop+=1
        current_loss=train_loss
 
        model.eval()
        auc_test_all=[]
        for user_features, user_labels in testing_loader:
            user_features = user_features.cuda()
            user_labels = user_labels.numpy()
            get_scores = model.prediction(user_features)
            pre_scores=get_scores.cpu().numpy() 
            y = (user_labels).reshape(-1) 
            pred_auc = pre_scores#np.max(pre_scores,axis=-1)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred_auc, pos_label=1)
            auc_test = metrics.auc(fpr, tpr)

            auc_one=round(auc_test,4) 
            if auc_one<0.5:
                auc_one=1-auc_one
            str_f1='gender ,auc:'+str(round(auc_test,4))#+'  f1_micro:'+str(round(f1_micro,4))#+' auc:'+str(round(auc_test,4))
            if flag_stop>=2: 
                res_auc.append(auc_one) 

        str_print_evl=str_print_train+" epoch:"+str(epoch)+str_f1 
        print(str_print_evl)






