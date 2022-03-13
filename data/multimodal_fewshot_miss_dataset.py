from itertools import count
import os
import sys
sys.path.append("/data/luowei/MMIN")
import json
import random
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset




class MultimodalFewshotMissDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAP_config.json')))
        self.norm_method = opt.norm_method
        # load feature
        self.A_type = opt.A_type
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'A', f'{self.A_type}.h5'), 'r')
        if self.A_type == 'comparE':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE_mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()
        self.V_type = opt.V_type
        self.all_V = h5py.File(os.path.join(config['feature_root'], 'V', f'{self.V_type}.h5'), 'r')
        self.L_type = opt.L_type
        self.all_L = h5py.File(os.path.join(config['feature_root'], 'L', f'{self.L_type}.h5'), 'r')
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)

        # few shot训练集，每个类别取16个样本，4个类则共64个样本
        if set_name == 'trn':
            self.fewshot_num = 16
            self.class_num = 4
            all_class_fewshots_idx = []
            for label in range(0,self.class_num):
                fewshots_idx = np.where(self.label == label)[0][0 : self.fewshot_num] # 前16个样本
                all_class_fewshots_idx = np.append(all_class_fewshots_idx,fewshots_idx)
            self.label = self.label[all_class_fewshots_idx.astype('int64')] # 标签数组
            self.int2name = self.int2name[all_class_fewshots_idx.astype('int64')] # 文件名数组
       

        # make missing index
        if set_name != 'trn':           # val && tst
            self.missing_index = torch.tensor([
                [1,0,0], # AZZ
                [0,1,0], # ZVZ
                [0,0,1], # ZZL
                [1,1,0], # AVZ
                [1,0,1], # AZL
                [0,1,1], # ZVL
                # [1,1,1]  # AVL
            ] * len(self.label)).long()
            self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'] * len(self.label)
        else:                           # trn
            self.missing_index = [
                [1,0,0], # AZZ
                [0,1,0], # ZVZ
                [0,0,1], # ZZL
                [1,1,0], # AVZ
                [1,0,1], # AZL
                [0,1,1], # ZVL
                # [1,1,1]  # AVL
            ]
            self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
        
        # set collate function
        self.manual_collate_fn = True

    def __len__(self):
        return len(self.missing_index) if self.set_name != 'trn' else self.class_num * self.fewshot_num

    def __getitem__(self, index):
        if self.set_name != 'trn':
            # val && tst
            # 对于val和tst，按顺序获取到不同模态缺失情况的数据
            feat_idx = index // 6         # totally 6 missing types
            missing_index = self.missing_index[index]    
            miss_type = self.miss_type[index]
        else:
            # trn
            # 对于trn，每次getitem时missingType是随机的
            feat_idx = index
            rand = random.randint(0,5)
            missing_index = torch.tensor(self.missing_index[rand]).long() # 每次随机选取一个缺失情况，当epoch足够多时，就好像数据集是原来的6倍了。
            miss_type = self.miss_type[rand]
        
        int2name = self.int2name[feat_idx][0].decode()
        label = torch.tensor(self.label[feat_idx])
        
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        
        return {
            # trn 返回的是完整版本
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        } if self.set_name == 'trn' else{
            # val,tst 返回的是缺失版本
            'A_feat': A_feat * missing_index[0],  # 做一个mask运算，例如miss_index = [0,1,1]，那么A_feat = zero
            'V_feat': V_feat * missing_index[1],
            'L_feat': L_feat * missing_index[2],
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        }
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    # 每读取batch个样本，就调用collate_fn进行打包
    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        missing_index = torch.cat([sample['missing_index'].unsqueeze(0) for sample in batch], axis=0)
        miss_type = [sample['miss_type'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        }

# run 
# /data/luowei/anaconda3/envs/wav2clip_env/bin/python /data/luowei/MMIN/data/multimodal_fewshot_miss_dataset.py
if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'
    
    opt = test()
    a = MultimodalFewshotMissDataset(opt, set_name='val')

    # print('Reading from dataloader:')
    # x = [a[i] for i in range(16)]
    # print('each one:')
    # for i, _x in enumerate(x):
    #     print(_x['missing_index'], _x['miss_type'])
    #     # 不同模态的sequence length 和 feature dimension是不一样的
    #     print(_x['A_feat'].shape,_x['V_feat'].shape,_x['L_feat'].shape) # torch.Size([xx, 130]) torch.Size([50, 342]) torch.Size([22, 1024])


    print('dataset len:')
    print(len(a))

    # 计算不同缺失的频率
    miss_type_count_dict = {}
    all_miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
    for miss_type in all_miss_type:
        miss_type_count_dict[miss_type] = 0
    label_count_dict = {}
    for i in range(0,4):
        label_count_dict[i] = 0
    for eopch in range(0,10):
        for i,data in enumerate(a):
            miss_type,label =  data['miss_type'],data['label']
            miss_type_count_dict[miss_type] += 1
            label_count_dict[int(label)] += 1
    print(miss_type_count_dict)
    print(label_count_dict)
        
    
    