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
import filter_layer
import pc_age_train
import pc_gender_train

##last FM 360K
user_num=359347#user_size
item_num=292589#item_size 
factor_num=64
batch_size=2048*128*1#3600*00

dataset_base_path='../../data/lastfm'
saved_model_path='..'

run_id="gc0"
print(run_id)
dataset='lastfm'
path_save_base='./log/'+dataset+'/filter_lineargcn'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)
result_file=open(path_save_base+'/results.txt','w+')#('./log/results_gcmc.txt','w+')
result_file_hr_ndcg=open(path_save_base+'/results_rmase.txt','w+')#('./log/results_gcmc.txt','w+')
result_file_classification=open(path_save_base+'/results_clf.txt','w+')#('./log/results_gcmc.txt','w+')
copyfile('./FairGo_gcn_remove_com.py', path_save_base+'/FairGo_gcn_remove_com'+run_id+'.py')
copyfile('./filter_layer.py', path_save_base+'/filter_layer'+run_id+'.py')
copyfile('./data_utils.py', path_save_base+'/data_utils'+run_id+'.py')
copyfile('./pc_age_train.py', path_save_base+'/pc_age_train'+run_id+'.py')
copyfile('./pc_gender_train.py', path_save_base+'/pc_gender_train'+run_id+'.py') 


path_save_model_base='../Model/'+dataset+'/filter_lineargcn'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

training_ratings_dict,train_dict_count = np.load(dataset_base_path+'/data1t5/training_ratings_dict.npy',allow_pickle=True)  
testing_ratings_dict,test_dict_count = np.load(dataset_base_path+'/data1t5/testing_ratings_dict.npy',allow_pickle=True)
val_ratings_dict,val_dict_count = np.load(dataset_base_path+'/data1t5/val_ratings_dict.npy',allow_pickle=True)

training_u_i_set,training_i_u_set = np.load(dataset_base_path+'/data1t5/train_adj_set.npy',allow_pickle=True)

users_emb_gcn = np.load('./gcnModel/user_emb_epoch50.npy',allow_pickle=True)    
items_emb_gcn = np.load('./gcnModel/item_emb_epoch50.npy',allow_pickle=True)    

gender_features = np.load(dataset_base_path+'/data1t5/genders.npy',allow_pickle=True)
age_features = np.load(dataset_base_path+'/data1t5/ages.npy',allow_pickle=True)
# users_features = np.load(dataset_base_path+'/data1t5/users_features.npy',allow_pickle=True)
users_features = np.load(dataset_base_path+'/data1t5/users_features_2num.npy')


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
        self.users_features = torch.cuda.LongTensor(users_features)
        self.gcn_users_embedding0 = torch.cuda.FloatTensor(gcn_user_embs)
        self.gcn_items_embedding0 = torch.cuda.FloatTensor(gcn_item_embs)
        self.user_num = user_num
        # self.gcn = filter_layer.LineGCN(user_num, item_num, int(factor_num/2))
        
        self.sigm = nn.Sigmoid()
        self.mse_loss=nn.MSELoss()
        self.model_d1 = filter_layer.DisClfGender(factor_num,2,attribute='gender',use_cross_entropy=True)
        self.model_d2 = filter_layer.DisClfAge(factor_num,3,attribute='age',use_cross_entropy=True) 
        self.model_f1 = filter_layer.AttributeFilter(factor_num, attribute='gender')
        self.model_f2 = filter_layer.AttributeFilter(factor_num, attribute='age') 

    def forward(self, adj_pos1,user_batch,rating_batch,item_batch,flag_t):      
        #format of pos_seq or neg_seq:user_item_matrix,item_user_matrix,d_i_train,d_j_train
        adj_pos = copy.deepcopy(adj_pos1)
        gcn_users_embedding0 = self.gcn_users_embedding0
        gcn_items_embedding0 = self.gcn_items_embedding0 
        # gcn_users_embedding0,gcn_items_embedding0 = self.gcn(adj_pos[0],adj_pos[1],adj_pos[2],adj_pos[3])

        # filter gender, age
        user_f1_tmp = self.model_f1(gcn_users_embedding0[user_batch])
        user_f2_tmp = self.model_f2(gcn_users_embedding0[user_batch])  
        #binary mask
        # if flag_t==1:
        d_mask = [torch.randint(0,2,(1,)),torch.randint(0,2,(1,))] 
        d_mask = torch.cuda.FloatTensor(d_mask)#.cuda()
        sum_d_mask = d_mask[0]+d_mask[1]
        while sum_d_mask <= 0:# ensure at least one filter
            d_mask = [torch.randint(0,2,(1,)),torch.randint(0,2,(1,))] 
            d_mask = torch.cuda.FloatTensor(d_mask)
            sum_d_mask = d_mask[0]+d_mask[1]
        user_f_tmp = (d_mask[0]*user_f1_tmp+d_mask[1]*user_f2_tmp)/sum_d_mask

        lables_gender = self.users_features[:,0]
        d_loss1 = self.model_d1(user_f_tmp,lables_gender[user_batch])
        lables_age = self.users_features[:,1]
        d_loss2 = self.model_d2(user_f_tmp,lables_age[user_batch])

        # #local attribute
        item_f1_tmp = self.model_f1(gcn_items_embedding0)
        item_f2_tmp = self.model_f2(gcn_items_embedding0) 

        users_f1_local = torch.sparse.mm(adj_pos1[0], item_f1_tmp)
        users_f2_local = torch.sparse.mm(adj_pos1[0], item_f2_tmp) 
        user_f_local_tmp = (d_mask[0]*users_f1_local+d_mask[1]*users_f2_local)/sum_d_mask
        d_loss1_local = self.model_d1(user_f_local_tmp,lables_gender)
        d_loss2_local = self.model_d2(user_f_local_tmp,lables_age) 

        w_f=[2,1]#[3,1.5,0.5]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]
        d_loss = (d_mask[0]*d_loss1*w_f[0]+ d_mask[1]*d_loss2*w_f[1])/sum_d_mask
        d_loss_local = (d_mask[0]*d_loss1_local*w_f[0]+ d_mask[1]*d_loss2_local*w_f[1])#/sum_d_mask

        #L_R preference prediction loss.
        user_b = (user_f1_tmp+user_f2_tmp)/2.0
        item_person_f = (item_f1_tmp+item_f2_tmp)/2.0#gcn_items_embedding0#item_f1_tmp
        # user_b = F.embedding(user_batch,user_person_f)
        item_b = F.embedding(item_batch,item_person_f)
        prediction = (user_b * item_b).sum(dim=-1)
        loss_part = self.mse_loss(prediction,rating_batch)
        l2_regulization = 0.001*(user_b**2+item_b**2).sum(dim=-1)
        # loss_part= -((prediction_i - prediction_j).sigmoid().log().mean())
        loss_p_square=loss_part+l2_regulization.mean()
        
        # if d_loss>500:
        #     print('main process')
        #     pdb.set_trace()
        d_loss_all= 10*(d_loss+0.5*d_loss_local)#+1*d_loss_local #+1*d_loss1_local.cpu().numpy()
        g_loss_all= 0.1*loss_p_square - 1*d_loss_all
        g_d_loss_all = - 1*d_loss_all
        d_g_loss = [d_loss_all,g_loss_all,g_d_loss_all]

        return d_g_loss
    # Detach the return variables
    def embed(self, adj_pos1): 
        adj_pos = copy.deepcopy(adj_pos1)
        # gcn_users_embedding0,gcn_items_embedding0 = self.gcn(adj_pos[0],adj_pos[1],adj_pos[2],adj_pos[3])
        gcn_users_embedding0 = self.gcn_users_embedding0
        gcn_items_embedding0 = self.gcn_items_embedding0 
        fliter_u_emb1 = self.model_f1(gcn_users_embedding0)
        fliter_u_emb2 = self.model_f2(gcn_users_embedding0) 
        fliter_i_emb1 = self.model_f1(gcn_items_embedding0)
        fliter_i_emb2 = self.model_f2(gcn_items_embedding0)  
        return gcn_users_embedding0.detach(),fliter_u_emb1.detach(),fliter_u_emb2.detach(),gcn_items_embedding0.detach(),fliter_i_emb1.detach(),fliter_i_emb2.detach()
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
        batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset_loss = data_utils.BPRData(
        train_raing_dict=val_ratings_dict,is_training=False, data_set_count=val_dict_count)
val_loader_loss = DataLoader(val_dataset_loss,
        batch_size=batch_size, shuffle=False, num_workers=0)


######################################################## TRAINING #####################################

all_nodes_num=user_num+item_num 
print('--------training processing-------')
count, best_hr = 0, 0

model = InforMax(user_num, item_num, factor_num,users_features,users_emb_gcn,items_emb_gcn)
model=model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)#, betas=(0.5, 0.99))
# d1_optimizer = torch.optim.Adam(model.model_d1.parameters(), lr=0.005)
# f1_optimizer = torch.optim.Adam(model.model_f1.parameters(), lr=0.005)
# gcn_optimizer = torch.optim.Adam(model.gcn.parameters(), lr=0.005)
f_optimizer = torch.optim.Adam(list(model.model_f1.parameters()) + \
                            list(model.model_f2.parameters()),lr=0.001)
d_optimizer = torch.optim.Adam(list(model.model_d1.parameters()) + \
                            list(model.model_d2.parameters()) ,lr=0.005)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


for epoch in range(300):
    model.train()  
    start_time = time.time() 
    print('train data is end')
    loss_current = [[],[],[],[]]
    for i_d in range(5):
        for user_batch, rating_batch, item_batch in train_loader: 
            user_batch = user_batch.cuda()
            rating_batch = rating_batch.cuda()
            item_batch = item_batch.cuda()
            d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch,1)
            d_l,r_l,f_l = d_g_l_get
            loss_current[0].append(d_l.item()) 
            d_optimizer.zero_grad()
            d_l.backward()
            d_optimizer.step()

    for user_batch, rating_batch, item_batch in train_loader: 
        user_batch = user_batch.cuda()
        rating_batch = rating_batch.cuda()
        item_batch = item_batch.cuda()
        d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch,1)
        d_l,r_l,f_l  = d_g_l_get 
        loss_current[1].append(r_l.item())  
        f_optimizer.zero_grad()
        r_l.backward()
        f_optimizer.step()

    # d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch,1)
    # d_l,r_l,f_l = d_g_l_get 
    # loss_current[2].append(f_l.item())  
    # f_optimizer.zero_grad()
    # f_l.backward()
    # f_optimizer.step()


    elapsed_time = time.time() - start_time 
    train_loss_d = round(np.mean(loss_current[0]),4)#
    train_loss_f = round(np.mean(loss_current[1]),4)#
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))
    str_print_train=format(str_print_train,'<20')
    
    str_d_g_str='train_loss'
    # str_d_g_str+=' f:'+str(train_loss_f)+'='+str(train_loss_f_g)+' - '+str(train_loss_f_d)
    str_d_g_str+=' f:'+str(train_loss_f)#+'fd:'+str(train_loss_f_d)
    str_d_g_str+=' d:'+str(train_loss_d)# 
    str_d_g_str =format(str_d_g_str,'<30')
    str_print_train +=str_d_g_str#'  d_1:'+str()
    
    if epoch>0: 
        PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
        torch.save(model.state_dict(), PATH_model)
    loss_test_val=[[],[],[],[]]
    for user_batch, rating_batch, item_batch in val_loader_loss: 
        user_batch = user_batch.cuda()
        rating_batch = rating_batch.cuda()
        item_batch = item_batch.cuda()
        d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch,1)
        d_l,r_l,f_l = d_g_l_get
        # pdb.set_trace()
        loss_test_val[0].append(d_l.item())
        loss_test_val[1].append(r_l.item())  
    val_loss_d  = round(np.mean(loss_test_val[0]),4)#
    val_loss_f  = round(np.mean(loss_test_val[1]),4)#

    str_tv_loss='val_loss'
    str_tv_loss+=' f:'+str(val_loss_f) 
    str_tv_loss+=' d:'+str(val_loss_d)
    str_tv_loss =format(str_tv_loss,'<28')
    str_print_train += str_tv_loss
    print(run_id+'--train--',elapsed_time)
    print(str_print_train)

    result_file.write(str_print_train)
    result_file.write('\n') 
    result_file.flush() 
     

    model.eval()
    user_emb_get,f1_u_emb,f2_u_emb,item_emb_get,f1_i_emb,f2_i_emb= model.embed(copy.deepcopy(pos_adj))
    user_emb_o = user_emb_get.cpu().numpy()
    item_emb_o = item_emb_get.cpu().numpy()
    user_e_f1 = f1_u_emb.cpu().numpy()
    user_e_f2 = f2_u_emb.cpu().numpy() 
    item_e_f1 = f1_i_emb.cpu().numpy()
    item_e_f2 = f2_i_emb.cpu().numpy() 
    user_e=(user_e_f1+user_e_f2)/2.0
    item_e=(item_e_f1+item_e_f2)/2.0
    if epoch>0:
        PATH_model_u_f1=path_save_model_base+'/user_emb_epoch'+str(epoch)+'.npy'
        np.save(PATH_model_u_f1,user_emb_o)
        PATH_model_u_f1=path_save_model_base+'/user_emb_f1_epoch'+str(epoch)+'.npy'
        np.save(PATH_model_u_f1,user_e_f1)
        PATH_model_u_f2=path_save_model_base+'/user_emb_f2_epoch'+str(epoch)+'.npy'
        np.save(PATH_model_u_f2,user_e_f2) 
        PATH_model_i_f1=path_save_model_base+'/item_emb_epoch'+str(epoch)+'.npy'
        np.save(PATH_model_i_f1,item_emb_o)
        PATH_model_i_f1=path_save_model_base+'/item_emb_f1_epoch'+str(epoch)+'.npy'
        np.save(PATH_model_i_f1,item_e_f1)
        PATH_model_i_f2=path_save_model_base+'/item_emb_f2_epoch'+str(epoch)+'.npy'
        np.save(PATH_model_i_f2,item_e_f2) 


    pre_all = []
    label_all = []
    for pair_i in val_ratings_dict: 
        u_id, r_v, i_id = val_ratings_dict[pair_i]
        pre_get = np.sum(user_e[u_id]*item_e[i_id]) 
        pre_all.append(pre_get)
        label_all.append(r_v)
    r_test=rmse(np.array(pre_all),np.array(label_all))
    res_test=round(np.mean(r_test),4) 
    str_print_evl="val_rmse:"+str(res_test)
    str_print_evl =format(str_print_evl,'<16')
    
    pre_all = []
    label_all = []
    for pair_i in testing_ratings_dict: 
        u_id, r_v, i_id = testing_ratings_dict[pair_i]
        pre_get = np.sum(user_e[u_id]*item_e[i_id]) 
        pre_all.append(pre_get)
        label_all.append(r_v) 
    r_test=rmse(np.array(pre_all),np.array(label_all))
    res_test=round(np.mean(r_test),4)  
    str_print_evl+=format(" test_rmse:"+str(res_test),'<17')
    
    if epoch>3:
        auc_one,auc_res= pc_gender_train.clf_gender_all_pre(run_id,epoch,user_e,factor_num) 
        str_f1f1_res='\t gender auc:'+str(round(auc_one,4))+'\t'
        str_print_evl+=str_f1f1_res
        # for i_f1 in auc_res:
        #     str_print_evl+=str(round(i_f1,4))+' '

        f1p_one,f1r_one,f1res_p,f1res_r= pc_age_train.clf_age_all_pre(run_id,epoch,user_e,factor_num)
        f1micro_f1=(2*f1p_one*f1r_one)/(f1p_one+f1r_one) 
        str_f1f1_res='age f1:'+str(round(f1micro_f1,4))+'\t'
        str_print_evl+=str_f1f1_res
        # f1_lsit=(2*np.array(f1res_p)*np.array(f1res_r))/(np.array(f1res_p)+np.array(f1res_r))
        # for i_f1 in f1_lsit:
        #     str_print_evl+=str(round(i_f1,4))+' '
    
    print(str_print_evl)
    str_print_evl+='\n'
    result_file_hr_ndcg.write(str_print_train+' '+str_print_evl)
    result_file_hr_ndcg.flush()