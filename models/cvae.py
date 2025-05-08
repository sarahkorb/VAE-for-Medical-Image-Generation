import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from models.base_vae import BaseVAE

Tensor = TypeVar('Tensor')

class ConditionalVAE(BaseVAE):
    
    
    
    def __init__(self, 
                 in_channels: int, 
                 latent_dim: int, 
                 num_classes: int, # disease classes for mulitho
                 hidden_dims: List = None, 
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes 
        self.label_dim = num_classes  #multi-hot vector for each class

        current_in_channels = in_channels + num_classes 

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512,]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(current_in_channels, out_channels=h_dim, #in channel has extra dim for label
                               kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            current_in_channels = h_dim

        # same as before
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []
        
        # NEed to add number of classes to decoder input 
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], 
                                       hidden_dims[i+1],
                                       kernel_size=3, 
                                       stride=2, 
                                       padding=1, 
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], 
                                                hidden_dims[-1], 
                                                kernel_size=3, 
                                                stride=2, 
                                                padding=1, 
                                                output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=1, 
                                      kernel_size=3, padding=1),
                            nn.Tanh()
        )

    def encode(self, x: Tensor, y: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input image [B x C x H x W]
        :param y: (Tensor) Multi-hot label vector [B x num_classes]
        :return: (Tensor) List of latent codes
        """

        B, C, H, W = x.shape #get x dimensions 
        # Expand y to image shape and concat wih image so we have imags and labels

        # Expand y into image shape: [B, num_classes, H, W] (prep for concatenation)
        y_img = y.view(B, self.num_classes, 1, 1).expand(B, self.num_classes, H, W)
        x_cond = torch.cat([x, y_img], dim=1) #get new x with class labels
        
        result = self.encoder(x_cond)

        result = torch.flatten(result, start_dim=1) # shape shoudl be [B, hidden_dim*4]
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :param y: [B, num_classes]
        :return: (Tensor) [B x C x H x W]
        """
        # Concatenate latent vector and label y
        z_cond = torch.cat([z, y], dim=1)  # [B, latent_dim + num_classes]
        result = self.decoder_input(z_cond)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, y: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, y), x, mu, log_var]

    def loss_function(self, 
                      *args, 
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, 
               num_samples: int, 
               current_device: int, 
               y: Tensor = None) -> Tensor:
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        if y is None:
            y = torch.randint(0, 2, (num_samples, self.num_classes)).float().to(current_device)
        return self.decode(z, y)

    def generate(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:

        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :param y: [B, num_classes]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, y)[0]

