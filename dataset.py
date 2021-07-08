import os
from utils import *
from torch.utils.data import Dataset


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
        img = load_image(img_path)
        return img, self.data[item]['class_id']
