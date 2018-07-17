import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from fgnet import FGNet

images_path = "C:/data/FGNET/FGNET/root/Aligned_images"

dataset = FGNet(images_dir = images_path, transforms = 
            transforms.Compose([transforms.ToTensor()]))

loader = DataLoader(dataset = dataset, batch_size = 10, shuffle=False)
assert isinstance(loader, DataLoader)
assert "check if all loaded", loader.dataset.count == len(os.listdir(images_path))
