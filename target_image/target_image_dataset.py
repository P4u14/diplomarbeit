import torch
from torch.utils.data import Dataset


class TargetImageDataset(Dataset):
    """
    PyTorch Dataset for a list of TargetImage objects.
    Provides access to preprocessed images and their file paths for use in DataLoader pipelines.

    Args:
        target_images (list): List of TargetImage objects.
    """

    def __init__(self, target_images):
        """
        Initialize the TargetImageDataset.
        Args:
            target_images (list): List of TargetImage objects.
        """
        self.target_images = target_images

    def __len__(self):
        """
        Return the number of images in the dataset.
        Returns:
            int: Number of TargetImage objects.
        """
        return len(self.target_images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: Dictionary with keys 'tensor' (image as torch.Tensor) and 'image_path' (str).
        """
        ti = self.target_images[idx]
        x = ti.preprocessed_image
        if torch.is_tensor(x) is False:
            x = torch.from_numpy(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,H,W)
        return {"tensor": x, "image_path": ti.image_path}
