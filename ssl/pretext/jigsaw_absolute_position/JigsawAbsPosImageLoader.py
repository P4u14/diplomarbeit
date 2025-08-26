# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017
@author: Biagio Brattoli
Modified by: Paula Schreiber
"""
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class DataLoader(data.Dataset):
    def __init__(self, data_path, txt_list, classes=9):
        self.data_path = data_path
        self.names, _ = self.__dataset_info(txt_list)
        self.N = len(self.names)
        self.classes = classes

        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]

        img = Image.open(framename).convert('RGB')
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        grid_size = int(np.sqrt(self.classes))
        if grid_size * grid_size != self.classes:
            raise ValueError("Die Anzahl der Classes (Patches) muss eine perfekte Quadratzahl sein (z.B. 9, 16, 25).")

        s = float(img.size[0]) / grid_size
        a = s / 2
        tiles = [None] * self.classes
        for n in range(self.classes):
            i = n // grid_size
            j = n % grid_size
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the classes (patches) independently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        middle_idx = int(self.classes / 2)  # The center tile index in the grid (e.g. for 9 patches, it's 4)
        other_indices = [i for i in range(self.classes) if i != middle_idx]
        np.random.shuffle(other_indices)

        # Create the shuffled order of tiles
        shuffled_indices = other_indices[:middle_idx] + [middle_idx] + other_indices[middle_idx:]

        # The ground truth `labels` should be the original positions of the tiles in the shuffled order.
        # We create a mapping from the shuffled index back to its original position.
        labels = np.zeros(self.classes, dtype=int)
        for new_pos, original_pos in enumerate(shuffled_indices):
            labels[new_pos] = original_pos

        data = [tiles[i] for i in shuffled_indices]
        data = torch.stack(data, 0)

        return data, torch.LongTensor(labels), tiles

    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes):
        # all_perm = np.load('permutations_%d.npy' % (classes))
        # The path below is adapted to the new permutation file.
        # It assumes the 'max' selection method was used.
        path = 'permutations/permutations_hamming_max_%d.npy' % (classes)
        try:
            all_perm = np.load(path)
        except FileNotFoundError:
            print(f"Permutation file not found at {path}. Falling back to old path.")
            path = 'permutations_%d.npy' % (classes)
            all_perm = np.load(path)

        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')
