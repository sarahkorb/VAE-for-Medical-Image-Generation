import os
import math
import torch
from torch import optim
from models.base_vae import BaseVAE
# from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from typing import List, Callable, Union, Any, TypeVar, Tuple
from pathlib import Path
import numpy as np

import csv
Tensor = TypeVar('Tensor')

class CVAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(CVAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

            #forward pass requires input and labels 
    def forward(self, input: Tensor, labels: Tensor, **kwargs) -> Tensor:
        return self.model(input, labels, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    
    def save_individual_images(self, tensor_batch, folder_path, prefix):
        """
        Saves each image in a batch individually.
        
        Args:
            tensor_batch: Tensor of shape [B, C, H, W]
            folder_path: Directory to save images
            prefix: Filename prefix (e.g., "recon" or "sample")
        """
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(tensor_batch):
            save_path = folder_path / f"{prefix}_{idx:03d}.png"
            vutils.save_image(img, save_path, normalize=True)
        

    def sample_images(self):
        # Get sample reconstruction image     
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, test_label)
        recon_dir = Path(self.logger.log_dir) / "Reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)

        #RECONSTRUCTION GRID
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir, 
                                       "Reconstructions",
                                         f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        
        #INDIVIDUAL RECONSTRUCTIONS
        self.save_individual_images(recons.data, Path(self.logger.log_dir) / "Reconstructions", prefix=f"recon_epoch{self.current_epoch}")


        #SAMPLE GENERATION
        try:

            sample_dir = Path(self.logger.log_dir) / "Samples"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Use real test labels to generate
            sample_label = test_label[:144].to(self.curr_device)  # Ensure batch size is 144
            samples = self.model.sample(144,        #Get samples 
                                        self.curr_device, 
                                        y=sample_label)
            vutils.save_image(samples.cpu().data, #saving grid
                              os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        # Save individual images and labels
            indiv_dir = sample_dir / f"Epoch_{self.current_epoch}_individuals"
            indiv_dir.mkdir(parents=True, exist_ok=True)

            csv_path = indiv_dir / "labels.csv"
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename'] + [f'label_{i}' for i in range(self.model.num_classes)])

                for i in range(sample_label.size(0)):
                    img_path = indiv_dir / f"sample_{i}.png"
                    label_path = indiv_dir / f"sample_{i}_label.npy"

                    #saving mimages and labels 
                    vutils.save_image(samples[i], img_path, normalize=True)
                    label_vector = sample_label[i].cpu().numpy()
                    np.save(label_path, label_vector)

                    writer.writerow([img_path.name] + label_vector.tolist())

        except Exception as e:
            print(f"Sample generation failed: {e}")


    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        if self.params.get('scheduler_gamma') is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma=self.params['scheduler_gamma'])
            scheds.append(scheduler)
            return optims, scheds

        return optims
