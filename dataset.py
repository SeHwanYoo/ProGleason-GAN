import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from glob import glob

class SiCAPv2(Dataset):
    def change_transform(self,transform):
        self.transform=transform

    def transform_labels(self,dataframe, dictionary_classes):
        X = []
        y = []
        for i in range(len(dataframe)):
            X.append(dataframe.loc[i][0])
            # NC--> Class 1
            if (dataframe.loc[i][1] == 1):
                y.append(dictionary_classes['NC'])
            # G3--> Class 2
            elif (dataframe.loc[i][2] == 1):
                y.append(dictionary_classes['G3'])
            # G4--> Class 3
            elif (dataframe.loc[i][3] == 1):
                y.append(dictionary_classes['G4'])
            # G5--> Class 4
            elif (dataframe.loc[i][4] == 1):
                y.append(dictionary_classes['G5'])

            elif (dataframe.loc[i][5] == 1):
                y.append(dictionary_classes['G4C'])

        return np.array(X), np.array(y)

    def __init__(self, transform=None, shape=(512,512)):
        # self.dictionary={'NC':0, 'G3':1,'G4':2, 'G5': 3, 'G4C':2}
        # dataframe = pd.read_excel(csv_file)
        # self.image_names,self.labels=self.transform_labels(dataframe,self.dictionary)
        self.imgs_path = glob('/home/yoos-bii/Desktop/data_tct/prostate_output/*/*.png')
        # self.root_dir= root_dir
        self.transform=transform
        # self.dataframe=dataframe
        # self.shape=shape
        # self.image_list=[]
        # self.labels_list=[]
        # print('Loading images to memory: ')
        # for img_path in self.imgs_path: 
        # for position, element in enumerate(tqdm(self.image_names[:50])):
            # img_path = os.path.join(self.root_dir, element)
            


    def __len__(self):
        #return len(self.dataframe)
        # return 50
        return len(self.imgs_path)

    def __getitem__(self, idx):
        
        img_path = self.imgs_path[idx]
        
        image = Image.open(img_path)
        image = np.array(image)
        image = Image.fromarray(image, 'RGB')
        if (self.transform):
            image = self.transform(image)
            
        label = int(img_path.split('/')[-2])
        # y_label = torch.tensor(label)]

        return image, label