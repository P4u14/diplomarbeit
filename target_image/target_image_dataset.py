import torch
from torch.utils.data import Dataset


class TargetImageDataset(Dataset):

    def __init__(self, target_images):
        self.target_images = target_images

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        ti = self.target_images[idx]
        x = ti.preprocessed_image
        if torch.is_tensor(x) is False:
            x = torch.from_numpy(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,H,W)
        return  {"tensor": x, "image_path": ti.image_path}