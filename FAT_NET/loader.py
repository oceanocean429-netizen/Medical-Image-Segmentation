from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from PIL import Image
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
from utils import *


# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(isic_loader, self).__init__() # Use self.__init__() for clarity
        self.train = train
        
        # --- 1. Load Data ---
        if train:
          self.data = np.load(path_Data+'data_train.npy')
          self.mask = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data = np.load(path_Data+'data_test.npy')
            self.mask = np.load(path_Data+'mask_test.npy')
          else:
            self.data = np.load(path_Data+'data_val.npy')
            self.mask = np.load(path_Data+'mask_val.npy')

        # --- 2. Normalize Data ---
        self.data = dataset_normalized(self.data)
        
        # --- 3. FIX: Collapse 2-Channel Mask to 1-Channel ---
        # Original mask shape: [N, 512, 512, 2]
        # We assume the foreground (lesion) is in the second channel (index 1)
        # We select only the foreground channel to match the model's 1-channel prediction.
        if self.mask.ndim == 4 and self.mask.shape[-1] == 2:
            self.mask = self.mask[:, :, :, 1]
        # self.mask = self.mask[:, :, :, 1] # Now shape is [N, 256, 256]
        
        # --- 4. Preprocessing (Re-add channel dim for consistency and normalize) ---
        self.mask = np.expand_dims(self.mask, axis=3) # Shape is [N, 256, 256, 1]
        if self.mask.max() > 1:
            self.mask = self.mask / 255.0
        # self.mask = self.mask/255. # This should normalize 0/255 to 0/1

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
        
        seg = torch.tensor(seg.copy(), dtype=torch.float32) # Use float32 for masks/targets
        img = torch.tensor(img.copy(), dtype=torch.float32) # Use float32 for input data
        
        # Rearrange from HWC (2,0,1) to CHW (Channel, Height, Width)
        img = img.permute(2, 0, 1) # [3, 512, 512]
        seg = seg.permute(2, 0, 1) # [1, 512, 512]

        return img, seg
    
    def random_rot_flip(self,image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self,image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


               
    def __len__(self):
        return len(self.data)
    