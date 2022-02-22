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

    # 训练参数
    parser.add_argument('--epoch',type=int,default=10,help='training epoch number')

    # log
    parser.add_argument('--log_dir',type=str,default='./lw/logs',help='where log saves')
    parser.add_argument('--log_filename',type=str,default='train_miss_transformer_lw',help='log filename')

    opt = parser.parse_args()
    return opt

def get_logger(path,suffix):
    logger = logging.getLogger('train_miss_transformer')
    logger.setLevel(logging.INFO)

    # handler1
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )   
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    cur_time = beijing_now.strftime('%Y-%m-%d-%H.%M.%S')
    file_handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # handler2
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler) # 输出到文件的handler
    logger.addHandler(console_handler) # # 输出到控制台的handler

    return logger

def eval(model,val_dataset):
    model.eval() # 进入到eval模式
    total_pred = []
    total_label = []
    for _,data in enumerate(val_dataset):
        L_feat = data['L_feat'].cuda()
        A_feat = data['A_feat'].cuda()
        V_feat = data['V_feat'].cuda()
        
        logits , _ = model(L_feat,A_feat,V_feat)
        
        # 预测，标签
        pred = F.softmax(logits, dim=-1).argmax(dim=1).detach().cpu().numpy()
        label = data['label']

        total_pred.append(pred)
        total_label.append(label)

        
    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)

    model.train() # 恢复到train模式

    return acc

# 将3个模态的数据拼接直接送入transformer，然后连接MLP作输出长度为8的向量
# bash lw/train_transformer_lw.sh
if __name__ == '__main__':
    model = MULTModel()
    model.cuda()

    opt = get_opt()

    # logger 
    logger_path = os.path.join(opt.log_dir, opt.log_filename, 'cvNo'+str(opt.cvNo)) 
    if not os.path.exists(logger_path):
        os.makedirs(logger_path) # makedirs 创建多级目录
    suffix = '_'.join([opt.model, opt.dataset_mode])
    logger = get_logger(logger_path,suffix)
    

    # dataset
    train_dataset,val_dataset = create_dataset_with_args(opt, set_name=['trn','val'])

    # optimizer
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=1e-5)

    # criterion
    criterion = getattr(nn, 'CrossEntropyLoss')()
    
    for epoch in range(opt.epoch):
        model.train()
        proc_loss = 0.0
        proc_size = 0
        for batch_id, data in enumerate(train_dataset):
            model.zero_grad()
            L_feat = data['L_feat'].cuda()
            A_feat = data['A_feat'].cuda()
            V_feat = data['V_feat'].cuda()
            label = data['label'].cuda()

            preds, _ = model(L_feat,A_feat,V_feat)

            # batch loss
            loss = criterion(preds,label)
            loss.backward()
            logger.info('epoch {}'.format(epoch + 1) + ' batch_id {}'.format(batch_id) +  ' batch_loss ' +  '{:.4f}'.format(loss))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()

            # overall loss
            proc_loss += loss.item() 
            proc_size += 1
            avg_loss = proc_loss / proc_size
        logger.info('epoch {}'.format(epoch+1) + ' epoch_loss ' +  '{:.4f}'.format(avg_loss))
        
        logger.info('start to eval...')
        acc = eval(model,val_dataset)
        logger.info('epoch {}'.format(epoch+1) + ' val_acc ' +  '{:.4f}'.format(acc))
    


