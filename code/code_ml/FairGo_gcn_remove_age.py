# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))
# print('0000') 
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.autograd as autograd 

from sklearn import metrics
from sklearn.metrics import f1_score
import pdb
import copy 
from collections import defaultdict
import time
import data_utils 
from shutil import copyfile
import pickle
# import layers#LineGCN, AvgReadout, Discriminator
import filter_layer 


##movieLens-1M
user_num=6040#user_size
item_num=3952#item_size 
factor_num=64
batch_size=2048*100
top_k=20
num_negative_test_val=-1##all

dataset_base_path='../../data/ml1m'
saved_model_path='..'  

run_id="ga0"
print(run_id)
dataset='movieLens-1M' 

training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/datanpy/val_set.npy',allow_pickle=True)    
user_rating_set_all,_,_ = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True)

training_ratings_dict,train_dict_count = np.load(dataset_base_path+'/data1t5/training_ratings_dict.npy',allow_pickle=True)  
testing_ratings_dict,test_dict_count = np.load(dataset_base_path+'/data1t5/testing_ratings_dict.npy',allow_pickle=True)  

training_u_i_set,training_i_u_set = np.load(dataset_base_path+'/data1t5/training_adj_set.npy',allow_pickle=True)

users_emb_gcn = np.load('./gcnModel/user_emb_epoch79.npy',allow_pickle=True)    
items_emb_gcn = np.load('./gcnModel/item_emb_epoch79.npy',allow_pickle=True)    

users_features=np.load(dataset_base_path+'/data1t5/users_features_3num.npy')


class InforMax(nn.Module):
    def __init__(self, user_num, item_num, factor_num,users_features,gcn_user_embs,gcn_item_embs):
        super(InforMax, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator
        """
        # self.gcn = filter_layer.AttributeLineGCN(user_num, item_num, factor_num,users_features,items_features)
        # pdb.set_trace() 
        self.users_features = torch.cuda.LongTensor(users_features)
        self.gcn_users_embedding0 = torch.cuda.FloatTensor(gcn_user_embs)
        self.gcn_items_embedding0 = torch.cuda.FloatTensor(gcn_item_embs)
        self.user_num = user_num
        
        self.sigm = nn.Sigmoid()
        self.mse_loss=nn.MSELoss()
        self.model_d1 = filter_layer.DisClfGender(factor_num,2,attribute='gender',use_cross_entropy=True)
        self.model_d2 = filter_layer.DisClfAge(factor_num,7,attribute='age',use_cross_entropy=True)
        self.model_d3 = filter_layer.DisClfOcc(factor_num,21,attribute='occupation',use_cross_entropy=True)
        
        self.model_f1 = filter_layer.AttributeFilter(factor_num, attribute='gender')
        self.model_f2 = filter_layer.AttributeFilter(factor_num, attribute='age')
        self.model_f3 = filter_layer.AttributeFilter(factor_num, attribute='occupation')

        # Adversarial ground truths
        self.real_d = torch.ones(user_num, 1).cuda()#Variable(torch.Tensor(user_num).fill_(1.0), requires_grad=False)
        self.fake_d = torch.zeros(user_num, 1).cuda()#Variable(torch.Tensor(user_num).fill_(0.0), requires_grad=False)

    def forward(self, adj_pos,user_batch,rating_batch,item_batch):      
        #format of pos_seq or neg_seq:user_item_matrix,item_user_matrix,d_i_train,d_j_train
        adj_pos1 = copy.deepcopy(adj_pos)
        gcn_users_embedding0 = self.gcn_users_embedding0
        gcn_items_embedding0 = self.gcn_items_embedding0 

        # filter gender, age,occupation
        user_f1_tmp = self.model_f1(gcn_users_embedding0)
        user_f2_tmp = self.model_f2(gcn_users_embedding0)
        user_f3_tmp = self.model_f3(gcn_users_embedding0)
        #binary mask 
        d_mask = [torch.randint(0,2,(1,)),torch.randint(0,2,(1,)),torch.randint(0,2,(1,))] 
        d_mask = torch.cuda.FloatTensor(d_mask)#.cuda()
        sum_d_mask = d_mask[0]+d_mask[1]+d_mask[2] 
        while sum_d_mask <= 0:# ensure at least one filter
            d_mask = [torch.randint(0,2,(1,)),torch.randint(0,2,(1,)),torch.randint(0,2,(1,))] 
            d_mask = torch.cuda.FloatTensor(d_mask)
            sum_d_mask = d_mask[0]+d_mask[1]+d_mask[2]
        user_f_tmp = (d_mask[0]*user_f1_tmp+d_mask[1]*user_f2_tmp+d_mask[2]*user_f3_tmp)/sum_d_mask

        lables_gender = self.users_features[:,0]
        d_loss1 = self.model_d1(user_f_tmp,lables_gender)
        lables_age = self.users_features[:,1]
        d_loss2 = self.model_d2(user_f_tmp,lables_age)
        lables_occ = self.users_features[:,2]
        d_loss3 = self.model_d3(user_f_tmp,lables_occ)
        # pdb.set_trace()

        # #local attribute 
        item_f1_tmp = self.model_f1(gcn_items_embedding0)
        item_f2_tmp = self.model_f2(gcn_items_embedding0)
        item_f3_tmp = self.model_f3(gcn_items_embedding0)

        users_f1_local = torch.sparse.mm(adj_pos1[0], item_f1_tmp)
        users_f2_local = torch.sparse.mm(adj_pos1[0], item_f2_tmp)
        users_f3_local = torch.sparse.mm(adj_pos1[0], item_f3_tmp)
        user_f_local_tmp = (d_mask[0]*users_f1_local+d_mask[1]*users_f2_local+d_mask[2]*users_f3_local)/sum_d_mask
        # lables_gender = self.users_features[:,0]
        # lables_age = self.users_features[:,1]
        # lables_occ = self.users_features[:,2]
        d_loss1_local = self.model_d1(user_f_local_tmp,lables_gender)
        d_loss2_local = self.model_d2(user_f_local_tmp,lables_age)
        d_loss3_local = self.model_d3(user_f_local_tmp,lables_occ)

        w_f=[1,2,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[3,1.5,0.5]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]
        d_loss = (d_mask[0]*d_loss1*w_f[0]+ d_mask[1]*d_loss2*w_f[1] + d_mask[2]*d_loss3*w_f[2])#/sum_d_mask
        d_loss_local = (d_mask[0]*d_loss1_local*w_f[0]+ d_mask[1]*d_loss2_local*w_f[1] + d_mask[2]*d_loss3_local*w_f[2])#/sum_d_mask

        #L_R preference prediction loss.
        user_person_f = user_f2_tmp
        item_person_f = item_f2_tmp#gcn_items_embedding0#item_f2_tmp

        user_b = F.embedding(user_batch,user_person_f)
        item_b = F.embedding(item_batch,item_person_f)
        prediction = (user_b * item_b).sum(dim=-1)
        loss_part = self.mse_loss(prediction,rating_batch)
        l2_regulization = 0.01*(user_b**2+item_b**2).sum(dim=-1)
        # loss_part= -((prediction_i - prediction_j).sigmoid().log().mean())
        loss_p_square=loss_part+l2_regulization.mean()

        d_loss_all= 1*(d_loss+1*d_loss_local)#+1*d_loss_local #+1*d_loss1_local.cpu().numpy()
        g_loss_all= 10*loss_p_square #- 1*d_loss_all
        g_d_loss_all = - 1*d_loss_all
        d_g_loss = [d_loss_all,g_loss_all,g_d_loss_all]

        return d_g_loss
    # Detach the return variables
    def embed(self, adj_pos):
        # h_pos: cat gcn_users_embedding and gcn_items_embedding, dim =0 
        fliter_u_emb1 = self.model_f1(self.gcn_users_embedding0)
        fliter_u_emb2 = self.model_f2(self.gcn_users_embedding0)
        fliter_u_emb3 = self.model_f3(self.gcn_users_embedding0)
        fliter_i_emb1 = self.model_f1(self.gcn_items_embedding0)
        fliter_i_emb2 = self.model_f2(self.gcn_items_embedding0)
        fliter_i_emb3 = self.model_f3(self.gcn_items_embedding0) 
        # fliter_i_emb = self.gcn_items_embedding0
        return fliter_u_emb1.detach(),fliter_u_emb2.detach(),fliter_u_emb3.detach(),fliter_i_emb1.detach(),fliter_i_emb2.detach(),fliter_i_emb3.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))
 

g_adj= data_utils.generate_adj(training_u_i_set,training_i_u_set,user_num,item_num)
pos_adj=g_adj.generate_pos()

train_dataset = data_utils.BPRData(
        train_raing_dict=training_ratings_dict,is_training=True, data_set_count=train_dict_count)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)
  
testing_dataset_loss = data_utils.BPRData(
        train_raing_dict=testing_ratings_dict,is_training=False, data_set_count=test_dict_count)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=test_dict_count, shuffle=False, num_workers=0)



######################################################## TRAINING #####################################

all_nodes_num=user_num+item_num 
print('--------training processing-------')
count, best_hr = 0, 0
 
model = InforMax(user_num, item_num, factor_num,users_features,users_emb_gcn,items_emb_gcn)
model=model.to('cuda') 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99)) 
# d1_optimizer = torch.optim.Adam(model.model_d1.parameters(), lr=0.005)
# f1_optimizer = torch.optim.Adam(model.model_f1.parameters(), lr=0.005) 
# gcn_optimizer = torch.optim.Adam(model.gcn.parameters(), lr=0.005)
f_optimizer = torch.optim.Adam(list(model.model_f1.parameters()) + \
                            list(model.model_f2.parameters()) + \
                            list(model.model_f3.parameters()) ,lr=0.001)
d_optimizer = torch.optim.Adam(list(model.model_d1.parameters()) + \
                            list(model.model_d2.parameters()) + \
                            list(model.model_d3.parameters()) ,lr=0.001)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


for epoch in range(100):
    model.train()  
    start_time = time.time() 
    print('train data is  end')

    loss_current = [[],[],[],[]]

    for user_batch, rating_batch, item_batch in train_loader: 
        user_batch = user_batch.cuda()
        rating_batch = rating_batch.cuda()
        item_batch = item_batch.cuda()
        d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch)
        d_l,f_l,_ = d_g_l_get
        loss_current[2].append(d_l.item()) 
        d_optimizer.zero_grad()
        d_l.backward()
        d_optimizer.step()
    
    for user_batch, rating_batch, item_batch in train_loader: 
        user_batch = user_batch.cuda()
        rating_batch = rating_batch.cuda()
        item_batch = item_batch.cuda()
        d_g_l_get = model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch)
        # d_g_l_get = model(copy.deepcopy(pos_adj),copy.deepcopy(pos_adj),user,item_i, item_j) 
        _,f_l,d_l = d_g_l_get 
        loss_current[0].append(f_l.item()) 
        # loss_current[1].append(d_l.item())  
        f_optimizer.zero_grad()
        f_l.backward()
        f_optimizer.step()
        # continue

    d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch)
    _,f_l,d_l = d_g_l_get 
    loss_current[1].append(d_l.item())  
    f_optimizer.zero_grad()
    d_l.backward()
    f_optimizer.step()


    loss_current=np.array(loss_current)
    elapsed_time = time.time() - start_time
    # pdb.set_trace()
    train_loss_f = round(np.mean(loss_current[0]),4)#
    train_loss_f_d = round(np.mean(loss_current[1]),4)# 
    train_loss_d=round(np.mean(loss_current[2]),4)#
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))#+' train loss:'+str(train_loss)+'='+str(train_loss_part)+'+'
    
    str_d_g_str=' loss'
    # str_d_g_str+=' f:'+str(train_loss_f)+'='+str(train_loss_f_g)+' - '+str(train_loss_f_d)
    str_d_g_str+=' f:'+str(train_loss_f)+'fd:'+str(train_loss_f_d)
    str_d_g_str+='\td:'+str(train_loss_d)# 
    str_print_train +=str_d_g_str#'  d_1:'+str()
    print(run_id+'--train--',elapsed_time) 
    print(str_print_train)

    result_file.write(str_print_train)
    result_file.write('\n') 
    result_file.flush() 
    
    model.eval()

    f1_users_embedding,f2_users_embedding,f3_users_embedding,f1_i_emb,f2_i_emb,f3_i_emb= model.embed(copy.deepcopy(pos_adj)) 
    user_e_f2 = f2_users_embedding.cpu().numpy() 
    item_e_f2 = f2_i_emb.cpu().numpy() 

    user_e = user_e_f2
    item_e = item_e_f2#items_emb_gcn
    str_print_evl=''#'epoch:'+str(epoch) 
    pre_all = []
    label_all = []
    for pair_i in testing_ratings_dict: 
        u_id, r_v, i_id = testing_ratings_dict[pair_i]
        pre_get = np.sum(user_e[u_id]*item_e[i_id]) 
        pre_all.append(pre_get)
        label_all.append(r_v) 
    r_test=rmse(np.array(pre_all),np.array(label_all))
    res_test=round(np.mean(r_test),4)
    str_print_evl+="\trmse:"+str(res_test) 
    print(str_print_evl) 