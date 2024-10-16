import os
from PIL import Image
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted(os.listdir(root))

    def __len__(self):
        
        return len(self.files)


    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        combined_img = Image.open(img_path)
        
     
        width, height = combined_img.size
        

        img_A = combined_img.crop((0, 0, width // 2, height))
        img_B = combined_img.crop((width // 2, 0, width, height))
        
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            
        return img_A, img_B