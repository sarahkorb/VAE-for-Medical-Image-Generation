import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import pandas as pd
# import medmnist  #UNCOMMENT HERE IF NEEDED! 
# from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import json


import random
from torchvision import transforms
import matplotlib.pyplot as plt

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

class FemaleChestXrayDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        transform: Optional[Callable] = None,
        finding_vocab: Optional[List[str]] = None
    ):
        self.data_dir = Path(data_path) / "ChestXrays_Original"
        self.images_dir = self.data_dir / "images"
        
        self.transforms = transform

        # Load the appropriate CSV: "female_train.csv" or "female_test.csv"
        csv_file = f"female_{split}.csv"
        self.metadata = pd.read_csv(self.data_dir / csv_file)

        #Unique lables 
        self.finding_vocab = finding_vocab or [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
            'Pneumothorax', 'No Finding'
        ]
        self.finding_to_index = {label: i for i, label in enumerate(self.finding_vocab)}

        mapping_path = Path(data_path) / "finding_to_index.json" #save mapping for later! 
        if not mapping_path.exists():
            with open(mapping_path, 'w') as f:
                json.dump(self.finding_to_index, f, indent=4)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.images_dir / row["Image Index"]
        img = default_loader(img_path)

        if self.transforms:
            img = self.transforms(img)

        # Convert finding labels to multi-hot vector
        findings = str(row["Finding Labels"]).split('|')
        label_vec = torch.zeros(len(self.finding_vocab), dtype=torch.float32)
        for f in findings:
            if f in self.finding_to_index:
                label_vec[self.finding_to_index[f]] = 1.0

        return img, label_vec


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class ProstateGleasonDataset(Dataset):
    """
    Prostate Gleason Dataset for histopathology images.
    """
    def __init__(self, data_path: str, split: str, transform: Optional[Callable] = None):
        self.data_dir = Path(data_path) / "prostate-gleason-dataset"
        self.transforms = transform
        
        # Choose the correct metadata file based on the split
        metadata_path = self.data_dir / (split + '.csv')  # 'train.csv' or 'test.csv'
        metadata = pd.read_csv(metadata_path)

        # Store the metadata
        self.metadata = metadata

        # Set the directory for images based on the split
        self.images_dir = self.data_dir / split  # 'train' or 'test' directory

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Retrieve image info from the metadata
        img_info = self.metadata.iloc[idx]
        img_path = self.images_dir / img_info['image']  # Load from the correct directory
        img = default_loader(img_path)
        
        label = img_info['class']  # The class label

        if self.transforms:
            img = self.transforms(img)

        return img, label

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        data_name: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.data_name = data_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
        if self.data_name == "oxfordpets":
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(self.patch_size),
    #                                               transforms.Resize(self.patch_size),
                                                transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            
            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(self.patch_size),
    #                                             transforms.Resize(self.patch_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

            self.train_dataset = OxfordPets(
                self.data_dir,
                split='train',
                transform=train_transforms,
            )
            
            self.val_dataset = OxfordPets(
                self.data_dir,
                split='val',
                transform=val_transforms,
            )
        
      # =========================  CelebA Dataset  =========================
        if self.data_name == 'celeba':
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.CenterCrop(148),
                                                    transforms.Resize(self.patch_size),
                                                    transforms.ToTensor(),])
            
            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor(),])
            
            self.train_dataset = MyCelebA(
                self.data_dir,
                split='train',
                transform=train_transforms,
                download=False,
            )
            
            # Replace CelebA with your dataset
            self.val_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=val_transforms,
                download=False,
            )
      # ===============================================================
#       =========================  Prostate Gleason Dataset =========================
        if self.data_name == 'prostategleason':

            prostate_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.patch_size),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5),(0.5))
            ])
            
            # if stage == "train" or stage is None:
            self.train_dataset = ProstateGleasonDataset(
                self.data_dir,
                split='train',
                transform=prostate_transforms,
            )
            
            # if stage == "val" or stage is None:
            self.val_dataset = ProstateGleasonDataset(
                self.data_dir,
                split='test',
                transform=prostate_transforms,
            )
#       ===============================================================
      # =========================  ChestMNIST =========================
        # if self.data_name == 'chestmnist':
            
        #     info = INFO['chestmnist']
        #     DataClass = getattr(medmnist, info['python_class'])

        #     chest_transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.CenterCrop(28),
        #         transforms.Resize(28),
        #         transforms.Normalize(mean=[.5], std=[.5])
        #     ])

        #     self.train_dataset = DataClass(split='train', transform=chest_transforms, download=True)


        #     self.val_dataset = DataClass(split='test', transform=chest_transforms, download=True)
    #       ===============================================================
      # =========================  female chestxrays =========================
      # NEED TO ADD HERE/FIX?
        if self.data_name == 'femchestxrays':

            femchestxrays_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # if stage == "train" or stage is None:
            self.train_dataset = FemaleChestXrayDataset(
                self.data_dir,
                split='train',
                transform=femchestxrays_transforms,
            )
            
            # if stage == "val" or stage is None:
            self.val_dataset = FemaleChestXrayDataset(
                self.data_dir,
                split='test',
                transform=femchestxrays_transforms,
            )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     

### TESTING CHEST DATASET :

# # Define transform
# test_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])  # grayscale normalization
# ])

# # Initialize dataset
# dataset = FemaleChestXrayDataset(
#     data_path="Data",   # path above ChestXrays_Original
#     split="train",
#     transform=test_transform
# )

# # Test one sample (Selecitng one w/ multi label)
# img, label_vec = dataset[7]

# # Assume img is (C, H, W)
# unnorm_img = img * 0.5 + 0.5  # unnormalize to [0,1]
# # Convert to numpy format for matplotlib: (H, W, C)
# img_np = unnorm_img.permute(1, 2, 0).numpy()
# # If grayscale (C=1), squeeze the channel
# if img_np.shape[-1] == 1:
#     img_np = img_np.squeeze(-1)

# #Show image
# plt.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
# plt.title(f"Condition vector: {label_vec.nonzero(as_tuple=True)[0].tolist()}")
# plt.axis('off')
# plt.show()

# # Confirm shapes and lables
# print("Image shape:", img.shape)
# print("Label vector:", label_vec)