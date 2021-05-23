import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image


class Dataset(Dataset):
    def __init__(self, data_dir, csv_path, transform, test = False):
        column_names = ['BA','BO','F','L','R','T','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.X_train = tmp_df[['BA','BO','F','L','R','T']]
        self.Y_train = tmp_df['MOS']

        self.length = len(tmp_df)
        self.test = test

    def __getitem__(self, index):        

        path_BA = os.path.join(self.data_dir,self.X_train.iloc[index,0])
        path_BO = os.path.join(self.data_dir,self.X_train.iloc[index,1])
        path_F = os.path.join(self.data_dir,self.X_train.iloc[index,2])
        path_L = os.path.join(self.data_dir,self.X_train.iloc[index,3])
        path_R = os.path.join(self.data_dir,self.X_train.iloc[index,4])
        path_T = os.path.join(self.data_dir,self.X_train.iloc[index,5])

        img_BA = Image.open(path_BA)
        img_BA = img_BA.convert('RGB')

        img_BO = Image.open(path_BO)
        img_BO = img_BO.convert('RGB')

        img_F = Image.open(path_F)
        img_F = img_F.convert('RGB')

        img_L = Image.open(path_L)
        img_L = img_L.convert('RGB')

        img_R = Image.open(path_R)
        img_R = img_R.convert('RGB')

        img_T = Image.open(path_T)
        img_T = img_T.convert('RGB')   

        if self.transform is not None:
            img_BA = self.transform(img_BA)
            img_BO = self.transform(img_BO)
            img_F = self.transform(img_F)
            img_L = self.transform(img_L)
            img_R = self.transform(img_R)
            img_T = self.transform(img_T)

        y_mos = self.Y_train.iloc[index]
        
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img_BA, img_BO, img_F, img_L, img_R, img_T, y_label        


    def __len__(self):
        return self.length






