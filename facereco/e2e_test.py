import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from fgnet import FGNet

images_path = "C:/data/FGNET/FGNET/root/Aligned_images"


def test_dataset(images_path):
    
    dataset = FGNet(images_dir = images_path, transforms = 
            transforms.Compose([transforms.ToTensor()]))

    loader = DataLoader(dataset = dataset, batch_size = 10, shuffle=False)
    assert isinstance(loader, DataLoader)
    assert "check if all loaded", loader.dataset.count == len(os.listdir(images_path))
    print('test_dataset test done')


def test_loader(images_path):
    dataset = FGNet(images_dir = images_path, transforms = 
            transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(dataset = dataset, batch_size = 1, shuffle=False)
    print('batch_size',loader.batch_size)

    m,c,w,h = 0,3,224,224
    for i, (x,y) in enumerate(loader):
        m+=x.size()[0]
    assert (m,c,h,w) == (len(os.listdir(images_path)), c,h,w)
   
    print('test_loader test done')
    

  
test_loader(images_path)  
