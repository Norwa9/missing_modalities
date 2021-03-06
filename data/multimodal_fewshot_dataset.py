import os
import sys
sys.path.append("/data/luowei/MMIN")
# print(sys.path)
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class MultimodalFewshotDataset(BaseDataset):
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
        # in aligned dataset, you should move out Ses03M_impro03_M001
        # if 'Ses03M_impro03_M001' in self.int2name:
        #     idx = self.int2name.index('Ses03M_impro03_M001')
        #     self.int2name.pop(idx)
        #     self.label = np.delete(self.label, idx, axis=0)
        self.manual_collate_fn = True


        # ???????????????????????? by luowei 2022/03/13
        if set_name == 'trn':
            self.fewshot_num = 16
            self.class_num = 4
            all_class_fewshots_idx = []
            for label in range(0,self.class_num):
                fewshots_idx = np.where(self.label == label)[0][0 : self.fewshot_num] # ???16?????????
                all_class_fewshots_idx = np.append(all_class_fewshots_idx,fewshots_idx)
            self.label = self.label[all_class_fewshots_idx.astype('int64')] # ????????????
            self.int2name = self.int2name[all_class_fewshots_idx.astype('int64')] # ???????????????


    def __getitem__(self, index):
        int2name = self.int2name[index][0].decode()
        label = torch.tensor(self.label[index])
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # process L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        # ????????????????????????
        return len(self.label) if self.set_name != 'trn' else self.class_num * self.fewshot_num
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        # ????????????????????????????????????
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }

# /data/luowei/anaconda3/envs/wav2clip_env/bin/python /data/luowei/MMIN/data/multimodal_fewshot_dataset.py
if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'

    opt = test()
    print('Reading from dataset:')
    a = MultimodalFewshotDataset(opt, set_name='trn')

    print('dataset len:')
    print(len(a))

    # ???????????????????????????????????????
    label_count_dict = {}
    for i in range(0,4):
        label_count_dict[i] = 0

    for eopch in range(0,10): # 10 x 64 = 640 ?????????
        for i,data in enumerate(a):
            label =  data['label']
            label_count_dict[int(label)] += 1
    print(label_count_dict)