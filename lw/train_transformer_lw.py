import logging
import sys
sys.path.append("/data/luowei/MMIN")
import os
from lw.models import MULTModel
from data.multimodal_miss_dataset import MultimodalMissDataset
from data.multimodal_dataset import MultimodalDataset
from data import create_dataset, create_dataset_with_args
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import time
import logging
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_opt():
    parser = argparse.ArgumentParser()
    # dataset参数
    parser.add_argument('--model', type=str, default='transformer', help='cross attention transformer')
    parser.add_argument('--dataset_mode', type=str, default='iemocap', help='chooses how datasets are loaded. [iemocap, ami, mix]')
    parser.add_argument('--A_type', type=str, help='which audio feat to use')
    parser.add_argument('--V_type', type=str, help='which visual feat to use')
    parser.add_argument('--L_type', type=str, help='which lexical feat to use')
    parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
    parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
    parser.add_argument('--cvNo', type=int, help='which cross validation set')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    # train
    parser.add_argument('--epoch',type=int,default=10,help='training epoch number')

    # test
    parser.add_argument('--has_test', action='store_true', help='whether have test. for 10 fold there is test setting, but in 5 fold there is no test')

    # log
    parser.add_argument('--log_dir',type=str,default='./lw/logs',help='where log saves')
    parser.add_argument('--log_filename',type=str,default='train_miss_transformer_lw',help='log filename')
    parser.add_argument('--img_dir',type=str,default='./lw/imgs',help='changes in metrics of the training process ')

    # save
    parser.add_argument('--checkpoints_dir', type=str, default='./lw/checkpoints', help='models are saved here')

    opt = parser.parse_args()
    return opt

def get_logger(opt):
    # file_path
    logger_path = os.path.join(opt.log_dir, opt.log_filename, 'cvNo'+str(opt.cvNo))
    if not os.path.exists(logger_path):
        os.makedirs(logger_path) # makedirs 创建多级目录
    suffix = '_'.join([opt.model, opt.dataset_mode])
    epoch = opt.epoch
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    cur_time = beijing_now.strftime('%Y-%m-%d-%H.%M.%S')
    file_path = os.path.join(logger_path, f"{cur_time}_{suffix}_{epoch}epoch.log")

    # logger
    logger = logging.getLogger('train_miss_transformer')
    logger.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler) # 输出到文件的handler
    logger.addHandler(console_handler) # # 输出到控制台的handler

    return logger

def save_network(model,epoch,opt):
        save_filename = 'epoch%s_%s_%s.pth' % (epoch, opt.model,opt.dataset_mode)
        save_path = os.path.join(opt.checkpoints_dir, save_filename)

        torch.save(model.cpu().state_dict(), save_path)
        model.cuda()

def load_network(model,epoch,opt):
    load_filename = 'epoch%s_%s_%s.pth' % (epoch, opt.model,opt.dataset_mode)
    load_path = os.path.join(opt.checkpoints_dir, load_filename)
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)

def get_model_size(model):
	para_num = sum([p.numel() for p in model.parameters()])
	# para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
	para_size = para_num * 4 / 1024 / 1024
	return para_size

def unpack_data(input,isTrain):
    """从data字典中获取到3模态的feature"""
    acoustic = input['A_feat'].cuda()
    lexical = input['L_feat'].cuda()
    visual = input['V_feat'].cuda()
    if isTrain and ('missing_index' in input): # isTrain && dataset=mutimodal_miss
        # 根据源码，完整模态下，dataloader返回的feature是模态完整的，所以需要根据missing_index制造模态缺失的feature
        missing_index = input['missing_index'].long().cuda()
        # A modality
        A_miss_index = missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        A_miss = acoustic * A_miss_index
        # L modality
        L_miss_index = missing_index[:, 2].unsqueeze(1).unsqueeze(2)
        L_miss = lexical * L_miss_index
        # V modality
        V_miss_index = missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        V_miss = visual * V_miss_index
    else: # !isTrain || dataset=mutimodal
        A_miss = acoustic
        V_miss = visual
        L_miss = lexical
    return A_miss,V_miss,L_miss

def plot(opt,acc_list,uar_list):
    acc_x = np.arange(len(acc_list))
    uar_x = np.arange(len(uar_list))
    acc_l = plt.plot(acc_x,acc_list,'r',label='acc')
    uar_l = plt.plot(uar_x,uar_list,'g',label='uar')
    plt.xlabel('epoch')
    plt.ylabel('acc/uar')
    plt.legend()
    plt.show()
    
    # save
    img_dir = os.path.join(opt.img_dir)
    file_name = '_'.join([opt.model, opt.dataset_mode , 'cvNo'+str(opt.cvNo), str(opt.epoch)+'epoch'])
    save_path = os.path.join(img_dir, file_name)
    plt.savefig(save_path)



def eval(model,val_dataset):
    model.eval() # 进入到eval模式
    total_pred = []
    total_label = []
    for _,data in enumerate(val_dataset):
        A_miss,V_miss,L_miss = unpack_data(data,isTrain=False)
        logits , _ = model(L_miss,A_miss,V_miss)
        
        # 预测，标签
        pred = F.softmax(logits, dim=-1).argmax(dim=1).detach().cpu().numpy()
        label = data['label']

        total_pred.append(pred)
        total_label.append(label)

        
    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro') # Unweighted Average Recall : macro 即使unweighted

    model.train() # 恢复到train模式

    return acc,uar

# 将3个模态的数据拼接直接送入transformer，然后连接MLP作输出长度为8的向量
# bash lw/train_transformer_lw.sh
if __name__ == '__main__':
    model = MULTModel()
    model_size = get_model_size(model)
    print('模型大小:',model_size)
    # print(model)
    model.cuda()

    opt = get_opt()

    # logger 
    logger = get_logger(opt)
    

    # dataset
    train_dataset,val_dataset,test_dataset = create_dataset_with_args(opt, set_name=['trn','val','tst'])

    # optimizer
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=1e-5)

    # criterion
    criterion = getattr(nn, 'CrossEntropyLoss')()
    
     
    best_eval_uar = 0              # record the best eval UAR
    best_eval_epoch = -1           # record the best eval epoch
    acc_list = []
    uar_list = []
    for epoch in range(opt.epoch):
        model.train()
        proc_loss = 0.0
        proc_size = 0
        for batch_id, data in enumerate(train_dataset):
            model.zero_grad()
            label = data['label'].cuda()
            A_miss,V_miss,L_miss = unpack_data(data,isTrain=True)
            preds, _ = model(L_miss,A_miss,V_miss)

            # batch loss
            loss = criterion(preds,label)
            loss.backward()
            logger.info('epoch {}'.format(epoch) + ' batch_id {}'.format(batch_id) +  ' batch_loss ' +  '{:.4f}'.format(loss))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()

            # overall loss
            proc_loss += loss.item() 
            proc_size += 1
            avg_loss = proc_loss / proc_size
        logger.info('epoch {}'.format(epoch) + ' epoch_loss ' +  '{:.4f}'.format(avg_loss))
        
        # eval
        logger.info('start to eval...')
        acc,uar = eval(model,val_dataset) # uar : Unweighted Average Recall
        logger.info('epoch {}'.format(epoch) + ' val_acc ' +  '{:.4f}'.format(acc) + ' uar ' +  '{:.4f}'.format(uar))
        save_network(model,epoch,opt)
        acc_list.append(acc)
        uar_list.append(uar)

        if uar > best_eval_uar: # uar是recall_score
            best_eval_epoch = epoch
            best_eval_uar = uar

    # test
    if opt.has_test:
        logger.info('Best eval epoch %d' % best_eval_epoch)
        load_network(model,best_eval_epoch,opt)
        logger.info('start to test...')
        acc, uar = eval(model,test_dataset)
        logger.info('Test result acc %.4f uar %.4f' % (acc, uar))
    
    # plot
    plot(opt,acc_list,uar_list)    


        
    


