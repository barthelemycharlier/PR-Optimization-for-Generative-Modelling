# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']
    elif self.dist_type == 'truncated':
      self.mean, self.var = kwargs['mean'], 1.0
      self.trunc = kwargs['var']
    elif dist_type == 'mixture_normal':
      # Required kwargs: mus (K, D), sigma (float)
      self.mus   = kwargs['mus']        # (K, latent_dim) on device
      self.sigma = kwargs['sigma']      # float
      self.K     = self.mus.size(0)
      self.weights = torch.ones(self.K, device=self.mus.device) / self.K

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)
    elif self.dist_type == 'truncated':
      temp = torch.randn_like(self)
      valid = temp <= self.trunc
      while not valid.prod():
        temp = torch.randn_like(self)*(~valid) + temp*valid
        valid = temp <= self.trunc
        self.data = temp
    elif self.dist_type == 'mixture_normal':
            # expected self.shape = (batch_size, latent_dim)
            B = self.size(0)

            # sample component indices
            idx = torch.multinomial(self.weights, B, replacement=True)

            # means for chosen components
            base = self.mus[idx]

            # noise
            z = base + self.sigma * torch.randn_like(base)

            self.data = z




  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)
    return new_obj


  # Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False,z_var=1.0, trunc=False):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  if trunc:
    z_.init_distribution('truncated', mean=0, var=z_var)
  else:
    z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device,torch.float16 if fp16 else torch.float32)

  if fp16:
    z_ = z_.half()

  y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical',num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_

import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm
import csv
import math
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


from tqdm import tqdm

# def evaluate_model(
#     G, device,
#     latent_dim=100, num_samples=10000, batch_size=64,
#     epoch=0, version="Vanilla", lr=0.0002, epochs=100, base_log_dir="logs",
#     sample_grid = True, avg_D_loss=None, avg_G_loss=None
# ):
#     """
#     Evaluate GAN by generating and saving sample images, organized by hyperparameters.
#     """
#     G.eval()
#     fake_images = []

#     with torch.no_grad():
#         for _ in tqdm(range(math.ceil(num_samples / batch_size))):
#             z = torch.randn(batch_size, latent_dim, device=device)
#             fake = G(z).detach()
#             fake = fake.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
#             fake_images.append(fake)

#     fake_images = torch.cat(fake_images, dim=0)[:num_samples]

#     if sample_grid :
#         fig, axes = plt.subplots(8, 8, figsize=(8, 8))
#         for i, ax in enumerate(axes.flatten()):
#             if i < fake_images.shape[0]:
#                 img = fake_images[i].cpu().numpy().transpose(1, 2, 0)
#                 img = (img + 1) / 2
#                 ax.imshow(img.squeeze(), cmap='gray')
#                 ax.axis('off')
#         plt.suptitle(f"Generated Samples - Epoch {epoch}")
#         plt.show()

#     G.train()
#     return avg_D_loss, avg_G_loss



def evaluate_model(
    G, device,
    latent_dim=100, num_samples=10000, batch_size=64,
    epoch=0, version="Vanilla", lr=0.0002, epochs=100, base_log_dir="logs",
    sample_grid=True, avg_D_loss=None, avg_G_loss=None
):
    """
    Evaluate GAN by generating and saving sample images,
    now using mixture-of-Gaussians sampling for z.
    """
    G.eval()
    fake_images = []

    K = 15
    c = 0.2
    sigma = 1.0

    mus = torch.empty(K, latent_dim, device=device).uniform_(-c, c)
    weights = torch.ones(K, device=device) / K


    with torch.no_grad():
        for _ in tqdm(range(math.ceil(num_samples / batch_size))):

            # sample mixture components
            idx = torch.multinomial(weights, batch_size, replacement=True)
            base = mus[idx]                                   # (B, latent_dim)

            # add noise
            z = base + sigma * torch.randn_like(base)

            # generate
            fake = G(z).detach()
            fake = fake.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
            fake_images.append(fake)

    fake_images = torch.cat(fake_images, dim=0)[:num_samples]

    if sample_grid:
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < fake_images.shape[0]:
                img = fake_images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2
                ax.imshow(img.squeeze(), cmap='gray')
                ax.axis('off')
        plt.suptitle(f"Generated Samples - Epoch {epoch}")
        plt.show()

    G.train()
    return avg_D_loss, avg_G_loss
