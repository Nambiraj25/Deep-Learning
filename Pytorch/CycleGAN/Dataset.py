
from PIL import Image
import os

import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self,root_A,root_B,transform=None):
    super().__init__()
    self.root_A=root_A
    self.root_B=root_B
    self.transform=transform

    self.img_A=os.listdir(root_A)
    self.img_B=os.listdir(root_B)
    self.length_dataset = max(len(self.img_A), len(self.img_B))

    self.img_A_len=len(self.img_A)
    self.img_B_len=len(self.img_B)
  def __len__(self):
    return self.length_dataset
  def __getitem__(self, index):
    A_img=self.img_A[index % self.img_A_len]
    B_img=self.img_B[index % self.img_B_len]


    path_A=os.path.join(self.root_A,A_img)
    path_B=os.path.join(self.root_B,B_img)


    A_img=Image.open(path_A).convert("RGB")
    B_img=Image.open(path_B).convert("RGB")

    if self.transform:
       A_img = self.transform(A_img)
       B_img = self.transform(B_img)
    return A_img,B_img

