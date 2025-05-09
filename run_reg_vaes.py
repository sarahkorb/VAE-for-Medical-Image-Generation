import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from models import vanilla_vae
from models import mssim_vae
from models import cvae
from experiment import VAEXperiment
from cond_experiment import CVAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy

#Change these config parameters accordingly
config = {
    'VanillaVAE': {
        'model_params': {
            'name': 'VanillaVAE',
            'in_channels': 3,
            'latent_dim': 128
        },
        'data_params': {
            'data_path': "Data/",
            'train_batch_size': 64,
            'val_batch_size':  64,
            'patch_size': 64,
            'data_name': 'prostategleason',
            'num_workers': 4,
        },
        'exp_params': {
            'LR': 0.005,
            'weight_decay': 0.0,
            'scheduler_gamma': 0.95,
            'kld_weight': 0.00025,
            'manual_seed': 1265
        },
        'trainer_params': {
            'max_epochs': 2 #changed!!
        },
        'logging_params': {
            'save_dir': "logs/",
            'name': "VanillaVAE"      
        }

    },
    'MSSIMVAE': {
        'model_params': {
            'name': 'MSSIMVAE',
            'in_channels': 3,
            'latent_dim': 128
        },
        'data_params': {
            'data_path': "Data/",
            'train_batch_size': 64,
            'val_batch_size':  64,
            'patch_size': 64,
            'data_name': 'prostategleason',
            'num_workers': 4,
        },
        'exp_params': {
            'LR': 0.005,
            'weight_decay': 0.0,
            'scheduler_gamma': 0.95,
            'kld_weight': 0.00025,
            'manual_seed': 1265
        },
        'trainer_params': {
            'max_epochs': 10
        },
        'logging_params': {
            'save_dir': "logs/",
            'name': "MSSIMVAE"      
        }
        },
    'CVAE': {
        'model_params': {
            'name': 'CVAE',
            'in_channels': 3,
            'latent_dim': 128,
            'num_classes': 15 
        },
        'data_params': {
            'data_path': "/content/drive/MyDrive/NNDL/Project/Image_Data/", #Change for Colab!
            'train_batch_size': 64, #change to 64 for GPU
            'val_batch_size':  64,
            'patch_size': 64,
            'data_name': 'femchestxrays',
            'num_workers': 0, #change to 4?
        },
        'exp_params': {
            'LR': 0.005,
            'weight_decay': 0.0,
            'scheduler_gamma': 0.95,
            'kld_weight': 0.00025,
            'manual_seed': 1265
        },
        'trainer_params': {
            'max_epochs': 15 #for now
        },
        'logging_params': {
            'save_dir': "logs/",
            'name': "CVAE"      
        }
        }
    }

models = {
    'VanillaVAE': vanilla_vae.VanillaVAE,
    'MSSIMVAE': mssim_vae.MSSIMVAE,
    'CVAE': cvae.ConditionalVAE
}

#CHANGE MODEL TYPE HERE
# MODEL = 'VanillaVAE'
MODEL = 'CVAE'

# Initializing logger
tb_logger =  TensorBoardLogger(save_dir=config[MODEL]['logging_params']['save_dir'],
                               name=config[MODEL]['model_params']['name'],)

# For reproducibility
seed_everything(config[MODEL]['exp_params']['manual_seed'], True)

model = models[config[MODEL]['model_params']['name']](**config[MODEL]['model_params'])

if MODEL == 'CVAE':
    experiment = CVAEXperiment(model,
                            config[MODEL]['exp_params'])
else:
    experiment = VAEXperiment(model,
                        config[MODEL]['exp_params'])

# Setting up PyTorch Lightning Datamodule object
data = VAEDataset(**config[MODEL]["data_params"], pin_memory=True)

data.setup()

# Initializing trainer object
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 accelerator='gpu', #remove for COLAB!!! 
                 devices=1,
                 strategy='ddp_notebook',
                #  strategy='auto',
                 **config[MODEL]['trainer_params'])


# Samples and reconstructions logged to Google Drive
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

# Fitting trainer object
print(f"======= Training {config[MODEL]['model_params']['name']} =======")
runner.fit(experiment, data)


