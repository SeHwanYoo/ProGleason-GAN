import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


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

    def __init__(self,csv_file,root_dir,transform=None,shape=(512,512)):
        self.dictionary={'NC':0, 'G3':1,'G4':2, 'G5': 3, 'G4C':2}
        dataframe = pd.read_excel(csv_file)
        self.image_names,self.labels=self.transform_labels(dataframe,self.dictionary)
        self.root_dir= root_dir
        self.transform=transform
        self.dataframe=dataframe
        self.shape=shape
        self.image_list=[]
        self.labels_list=[]
        print('Loading images to memory: ')
        for position, element in enumerate(tqdm(self.image_names[:50])):
            img_path = os.path.join(self.root_dir, element)
            image = Image.open(img_path)
            image = np.array(image)
            image = Image.fromarray(image, 'RGB')
            if (self.transform):
                image = self.transform(image)
            y_label = torch.tensor(self.labels[position])
            self.image_list.append(image)
            self.labels_list.append(y_label)


    def __len__(self):
        #return len(self.dataframe)
        return 50

    def __getitem__(self, item):
        image = self.transform(self.image_list[item])
        y_label=self.labels_list[item]

        return image,y_label