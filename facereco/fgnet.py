
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FGNet(Dataset):
    """
        Dataset class for FGNet dataset where dataset is arranged as follow: ::
             images_dir/001A02.JPG
             images_dir/001A03.JPG
             images_dir/001A04.JPG
    """

    def __init__(self, images_dir ,transforms = None):
        
        if not os.path.isdir(images_dir) or len(os.listdir(images_dir)) == 0:
            assert("invalid/empty directory ", images_dir)

        self.metadata = self.__load_info(images_dir)
        self.images_path = self.metadata.iloc[:,0]
        self.classes = self.metadata.iloc[:,2]
        
        self.count = len(self.images_path)
        self.transform = transforms

    def __load_info(self,path):
        
        '''
        Generate metadata dataframe from FGnet images directory

        parameters:
            path: directory path to Images directory
        
        return:
             pandas dataframe with schema "Image full path, age, label"
        '''
        df = pd.DataFrame(columns=['path', 'age','label'])

        for  file in os.listdir(path):
            
            im_path = os.path.join(path,file)
            index = file.find('A')
            age = int(file[index+1:index+3])
            label = int(file[:index])
            
            df = df.append(pd.Series([im_path,age,label],index=['path', 'age','label']), ignore_index=True)
        
        return df  
    
    
    def __getitem__(self, index):
    
        image = Image.open(self.images_path[index])
        if self.transform is not None:
            image = self.transform(image)

        return (image, self.classes[index])

    def __len__(self):
        return self.count 