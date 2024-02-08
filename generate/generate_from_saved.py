import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
########################
import torch
import dnnlib
import legacy
from training import    loss
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torchvision.utils import save_image

from torch.utils.data import Subset # DELETE AFTER
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from IPython.display import HTML
#########################################
from torch.autograd import Variable
from torch import autograd
from torchvision import utils


network_pkl = "D:/Flask_images/generate/generator_4000.pkl"
#device = torch.device('cpu')
#Load model
#with dnnlib.util.open_url(network_pkl) as f:
#   data = legacy.load_network_pkl(f)

Generator_Kwargs= {'z_dim': 512, 'c_dim': 0, 'w_dim': 512, 'img_resolution': 256, 'img_channels': 3, 'mapping_kwargs': {'num_layers': 8, 'embed_features': None, 'layer_features': None, 'activation': 'lrelu', 'lr_multiplier': 0.01, 'w_avg_beta': 0.995}, 'synthesis_kwargs': {'channel_base': 16384, 'channel_max': 512, 'num_fp16_res': 0, 'conv_clamp': None, 'architecture': 'skip', 'resample_filter': [1, 3, 3, 1], 'use_noise': True, 'activation': 'lrelu'}}

Discriminator_Kwargs= {'c_dim': 0, 'img_resolution': 256, 'img_channels': 3, 'architecture': 'resnet', 'channel_base': 16384, 'channel_max': 512, 'num_fp16_res': 0, 'conv_clamp': None, 'cmap_dim': None, 'block_kwargs': {'activation': 'lrelu', 'resample_filter': [1, 3, 3, 1], 'freeze_layers': 0}, 'mapping_kwargs': {'num_layers': 0, 'embed_features': None, 'layer_features': None, 'activation': 'lrelu', 'lr_multiplier': 0.1}, 'epilogue_kwargs': {'mbstd_group_size': None, 'mbstd_num_channels': 1, 'activation': 'lrelu'}}

ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

from training import networks
generator = networks.Generator(Generator_Kwargs['z_dim'], Generator_Kwargs['c_dim'], Generator_Kwargs['w_dim'], Generator_Kwargs['img_resolution'], Generator_Kwargs['img_channels'], Generator_Kwargs['mapping_kwargs'], Generator_Kwargs['synthesis_kwargs'])
data = torch.load(network_pkl)
generator.load_state_dict(data)

#generator.load_state_dict(data['G'].state_dict())
generator.to(device)


#discriminator = networks.Discriminator(Discriminator_Kwargs['c_dim'], Discriminator_Kwargs['img_resolution'], Discriminator_Kwargs['img_channels'], Discriminator_Kwargs['architecture'], Discriminator_Kwargs['channel_base'], Discriminator_Kwargs['channel_max'], Discriminator_Kwargs['num_fp16_res'], Discriminator_Kwargs['conv_clamp'], Discriminator_Kwargs['cmap_dim'], Discriminator_Kwargs['block_kwargs'], Discriminator_Kwargs['mapping_kwargs'], Discriminator_Kwargs['epilogue_kwargs'])
#discriminator.load_state_dict(data['D'].state_dict())
#iscriminator.to(device)

# Set random seed for reproducibility
#manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results

# Number of GPUs available. Use 0 for CPU mode.
#ngpu = 1

# Decide which device we want to run on
#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#  the progression of the generator
fixed_noise = noise = torch.randn(64, generator.z_dim, device=device)

#with torch.no_grad():
#    fake = generator(fixed_noise, None).detach().cpu()
            # fake = (fake - -1) * (255 / (1 - -1))
            #img_grid_fake = vutils.make_grid(fake, padding=2, normalize=True)
#img_grid_fake = vutils.make_grid(fake, padding=2)

            # Save the PyTorch tensor as an image
#save_image(img_grid_fake, f"lotest_wgantrue.png", normalize=True)
            #save_image(img_grid_fake, f"generated_images/epoch{epoch}.png")
with torch.no_grad():
    samples = generator(fixed_noise, 'Chrysanthemum')
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()[:18]
    grid = utils.make_grid(samples)
    utils.save_image(grid, 'chrys.png'.format(str(5).zfill(3)))
           
######################################
