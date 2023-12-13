#Because the training loop is different, running the vae-gan is done in a different file.
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
from utils import show_and_save, plot_loss
from torchvision.utils import make_grid , save_image
from dataset import VAEDataset
from models.vaegan import VAE_GAN, Discriminator
'''
Before Running this file, make sure that whatever dataset you are using is downloaded to the /Data directory. Also make sure to change the params of the config file appropriatly. 
'''
# Configuration for logging
config = {
    'logging_params': {
        'save_dir': 'runs/vae_gan_1'
    },
    #CHANGE THESE PARAMETERS ACCORDINGLY
    'data_params': {
        'data_path': 'Data/', # Change this to the correct path
        'train_batch_size': 64,
        'val_batch_size': 64,
        'data_name': 'prostategleason',
        'patch_size': 64 # this is essentially the size to which the images will be preprocessed
    }
}

# # Create the TensorBoard log directory
# log_dir = config['logging_params']['save_dir']
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/vae_gan_1')

def validate_model(val_loader, gen, discrim, criterion, device, gamma):
    gen.eval()
    discrim.eval()

    total_gan_loss = 0
    total_rec_loss = 0
    total_prior_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader, 0):
            bs = data.size()[0]

            ones_label=Variable(torch.ones(bs,1)).to(device)
            zeros_label=Variable(torch.zeros(bs,1)).to(device)
            zeros_label1=Variable(torch.zeros(64,1)).to(device)  # Check if this size is correct

            # Forward pass through the generator
            datav = Variable(data).to(device)
            mean, logvar, rec_enc = gen(datav)
            z_p = Variable(torch.randn(64,128)).to(device)
            x_p_tilda = gen.decoder(z_p)

            # Forward pass through the discriminator
            output = discrim(datav)[0]
            errD_real = criterion(output, ones_label)
            output = discrim(rec_enc)[0]
            errD_rec_enc = criterion(output, zeros_label)
            output = discrim(x_p_tilda)[0]
            errD_rec_noise = criterion(output, zeros_label1)

            # Calculate losses
            gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            x_l_tilda = discrim(rec_enc)[1]
            x_l = discrim(datav)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)

            # Accumulate the losses
            total_gan_loss += gan_loss.item()
            total_rec_loss += rec_loss.item()
            total_prior_loss += prior_loss.item()

    avg_gan_loss = total_gan_loss / len(val_loader)
    avg_rec_loss = total_rec_loss / len(val_loader)
    avg_prior_loss = total_prior_loss / len(val_loader)

    return avg_gan_loss, avg_rec_loss, avg_prior_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
print(config['data_params'])
data = VAEDataset(data_path=config['data_params']['data_path'], data_name=config['data_params']['data_name'], train_batch_size=config['data_params']['train_batch_size'], val_batch_size=config['data_params']['val_batch_size'], patch_size=config['data_params']['patch_size'], pin_memory=True)
data.setup()

data_loader=data.train_dataloader()
val_loader=data.val_dataloader()
gen=VAE_GAN().to(device)
discrim=Discriminator().to(device)
real_batch = next(iter(data_loader))
show_and_save("training" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))

epochs=10
lr=3e-4
alpha=0.1
gamma=15

criterion=nn.BCELoss().to(device)
optim_E=torch.optim.RMSprop(gen.encoder.parameters(), lr=lr)
optim_D=torch.optim.RMSprop(gen.decoder.parameters(), lr=lr)
optim_Dis=torch.optim.RMSprop(discrim.parameters(), lr=lr*alpha)
z_fixed=Variable(torch.randn((64,128))).to(device)
x_fixed=Variable(real_batch[0]).to(device)

progress_bar = tqdm(total=len(data_loader) * epochs, ncols=100, desc='Epoch: 0, Batch: 0/0')

for epoch in range(epochs):
  prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
  dis_real_list, dis_fake_list, dis_prior_list = [], [], []

  gen.train()
  discrim.train()

  for i, (data, _) in enumerate(data_loader, 0):
    progress_bar.set_description(f'Epoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(data_loader)}')
    bs = data.size()[0]

    ones_label=Variable(torch.ones(bs,1)).to(device)
    zeros_label=Variable(torch.zeros(bs,1)).to(device)
    zeros_label1=Variable(torch.zeros(64,1)).to(device)
    datav = Variable(data).to(device)
    mean, logvar, rec_enc = gen(datav)
    z_p = Variable(torch.randn(64,128)).to(device)
    x_p_tilda = gen.decoder(z_p)

    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    dis_real_list.append(errD_real.item())
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    dis_fake_list.append(errD_rec_enc.item())
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    dis_prior_list.append(errD_rec_noise.item())
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    gan_loss_list.append(gan_loss.item())
    optim_Dis.zero_grad()
    gan_loss.backward(retain_graph=True)
    optim_Dis.step()


    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise


    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    err_dec = gamma * rec_loss - gan_loss
    recon_loss_list.append(rec_loss.item())
    optim_D.zero_grad()
    err_dec.backward(retain_graph=True)
    optim_D.step()

    mean, logvar, rec_enc = gen(datav)
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
    prior_loss_list.append(prior_loss.item())
    err_enc = prior_loss + 5*rec_loss

    optim_E.zero_grad()
    err_enc.backward(retain_graph=True)
    optim_E.step()

    #if i % 50 == 0:
    #  progress_bar.write('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
    #        % (epoch, epochs, i, len(data_loader),
    #            gan_loss.item(), prior_loss.item(), rec_loss.item(), errD_real.item(), errD_rec_enc.item(), errD_rec_noise.item()))


    # Log losses to TensorBoard
    writer.add_scalar('Loss/GAN', gan_loss.item(), epoch * len(data_loader) + i)
    writer.add_scalar('Loss/Prior', prior_loss.item(), epoch * len(data_loader) + i)
    writer.add_scalar('Loss/Reconstruction', rec_loss.item(), epoch * len(data_loader) + i)
    writer.add_scalar('Loss/Discriminator_Real', errD_real.item(), epoch * len(data_loader) + i)
    writer.add_scalar('Loss/Discriminator_Fake', errD_rec_enc.item(), epoch * len(data_loader) + i)
    writer.add_scalar('Loss/Discriminator_Prior', errD_rec_noise.item(), epoch * len(data_loader) + i)

    # Update the progress bar
    progress_bar.update(1)



  val_gan_loss, val_rec_loss, val_prior_loss = validate_model(val_loader, gen, discrim, criterion, device, gamma)
  writer.add_scalar('Val Loss/GAN', val_gan_loss, epoch)
  writer.add_scalar('Val Loss/Reconstruction', val_rec_loss, epoch)
  writer.add_scalar('Val Loss/Prior', val_prior_loss, epoch)

  b=gen(x_fixed)[2]
  b=b.detach()
  c=gen.decoder(z_fixed)
  c=c.detach()
  show_and_save('prostate_noise_epoch_%d' % epoch ,make_grid((c*0.5+0.5).cpu(),8))
  show_and_save('prostate_rec_epoch_%d' % epoch ,make_grid((b*0.5+0.5).cpu(),8))

writer.close()
plot_loss(prior_loss_list)
plot_loss(recon_loss_list)
plot_loss(gan_loss_list)