import os
import cv2
import torch
from torch.utils.data import Dataset

IM_H = 128
IM_W = 128


class PokeDataset(Dataset):
    def __init__(self, root_dir, names):
        self.data = []
        self.root_dir = root_dir
        for i, pokemon in enumerate(names):
            for file in os.listdir(os.path.join(self.root_dir, pokemon)):
                self.data.append({'filename': file, 'name': pokemon, 'class_id': i})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.data[item]['name'], self.data[item]['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IM_H, IM_W))
        img = torch.tensor(img.transpose(2, 0, 1) / 255.0).float()

        return img, self.data[item]['class_id']
