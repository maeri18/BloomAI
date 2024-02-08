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

network_pkl = "/home/users/ichekam/data-efficient-gans/DiffAugment-stylegan2-pytorch/flowers-256-slim-001212.pkl"
device = torch.device('cuda')
#Load model
with dnnlib.util.open_url(network_pkl) as f:
    data = legacy.load_network_pkl(f)

Generator_Kwargs= {'z_dim': 512, 'c_dim': 0, 'w_dim': 512, 'img_resolution': 256, 'img_channels': 3, 'mapping_kwargs': {'num_layers': 8, 'embed_features': None, 'layer_features': None, 'activation': 'lrelu', 'lr_multiplier': 0.01, 'w_avg_beta': 0.995}, 'synthesis_kwargs': {'channel_base': 16384, 'channel_max': 512, 'num_fp16_res': 0, 'conv_clamp': None, 'architecture': 'skip', 'resample_filter': [1, 3, 3, 1], 'use_noise': True, 'activation': 'lrelu'}}

Discriminator_Kwargs= {'c_dim': 0, 'img_resolution': 256, 'img_channels': 3, 'architecture': 'resnet', 'channel_base': 16384, 'channel_max': 512, 'num_fp16_res': 0, 'conv_clamp': None, 'cmap_dim': None, 'block_kwargs': {'activation': 'lrelu', 'resample_filter': [1, 3, 3, 1], 'freeze_layers': 0}, 'mapping_kwargs': {'num_layers': 0, 'embed_features': None, 'layer_features': None, 'activation': 'lrelu', 'lr_multiplier': 0.1}, 'epilogue_kwargs': {'mbstd_group_size': None, 'mbstd_num_channels': 1, 'activation': 'lrelu'}}

from training import networks
generator = networks.Generator(Generator_Kwargs['z_dim'], Generator_Kwargs['c_dim'], Generator_Kwargs['w_dim'], Generator_Kwargs['img_resolution'], Generator_Kwargs['img_channels'], Generator_Kwargs['mapping_kwargs'], Generator_Kwargs['synthesis_kwargs'])
generator.load_state_dict(data['G'].state_dict())
generator.to(device)


discriminator = networks.Discriminator(Discriminator_Kwargs['c_dim'], Discriminator_Kwargs['img_resolution'], Discriminator_Kwargs['img_channels'], Discriminator_Kwargs['architecture'], Discriminator_Kwargs['channel_base'], Discriminator_Kwargs['channel_max'], Discriminator_Kwargs['num_fp16_res'], Discriminator_Kwargs['conv_clamp'], Discriminator_Kwargs['cmap_dim'], Discriminator_Kwargs['block_kwargs'], Discriminator_Kwargs['mapping_kwargs'], Discriminator_Kwargs['epilogue_kwargs'])
discriminator.load_state_dict(data['D'].state_dict())
discriminator.to(device)

# Set random seed for reproducibility
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True) # Needed for reproducible results


# Root directory for dataset
dataroot = "/home/users/ichekam/data-efficient-gans/DiffAugment-stylegan2-pytorch/flower_dataset/fa"

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 2000

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                           ]))
# So if you have mead=0 and std=1 then output=(output - 0) / 1 will not change.

# Define the size of the subset
#subset_size = 10000

# Create a random subset
#subset_indices = torch.randperm(len(dataset))[:subset_size]
#subset = Subset(dataset, subset_indices)
#dataset = subset

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images # LIBRARIES NOT INSTALLED
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()
#criterion = loss.StyleGAN2Loss(device, generator, generator, discriminator)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = noise = torch.randn(64, generator.z_dim, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        phase_real_c = data[1].to(device)

        # Check if any element is greater than 1
        # is_greater_than_1 = torch.any(real_cpu > 1)

        # Check if any element is smaller than 0
        # is_smaller_than_0 = torch.any(real_cpu < 0)

        # Print the results
        #print(f"Any element greater than 1: {is_greater_than_1}")
        #print(f"Any element smaller than 0: {is_smaller_than_0}")
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_cpu, label).view(-1)
        # Apply sigmoid to output **NEW**
        m = nn.Sigmoid()
        # Calculate loss on all-real batch
        errD_real = criterion(m(output), label)
        
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        ## Train with all-fake batch
        # Generate batch of latent vectors
        #noise = torch.randn(b_size, nz, 1, 1, device=device)
        noise = torch.randn(b_size, generator.z_dim, device=device)
        
        label.fill_(fake_label)
        
        # Generate fake image batch with G
        fake = generator(noise, label)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach(), label).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(m(output), label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake, label).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(m(output), label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        # Check how the generator is doing by saving G's output on fixed_noise
        #if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        if (epoch % 50== 0):
            with torch.no_grad():
                fake = generator(fixed_noise, label).detach().cpu()
            # fake = (fake - -1) * (255 / (1 - -1))
            #img_grid_fake = vutils.make_grid(fake, padding=2, normalize=True)
            img_grid_fake = vutils.make_grid(fake, padding=2)
            
            # Save the PyTorch tensor as an image
            save_image(img_grid_fake, f"fa_results/generated_images/lr2/epoch{epoch}.png", normalize=True)
            #save_image(img_grid_fake, f"generated_images/epoch{epoch}.png")
            img_list.append(img_grid_fake)

            # Save the pytorch model
            torch.save(generator, f"fa_results/generated_model/lr2/generator{epoch}.pt")

            torch.save(discriminator, f"fa_results/generated_model/lr2/discriminator{epoch}.pt")

        iters += 1
#######################################

