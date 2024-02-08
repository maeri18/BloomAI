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


###################################################
from training import networks
############################################
from torch.autograd import Variable
from torch import autograd
from torchvision import utils



#load the model
network_pkl = "/home/users/ichekam/data-efficient-gans/DiffAugment-stylegan2-pytorch/flowers-256-slim-001212.pkl"
device = torch.device('cuda')
cuda = True
cuda_index = 0
with dnnlib.util.open_url(network_pkl) as f:
    data = legacy.load_network_pkl(f)

#generator arguments
Generator_Kwargs= {'z_dim': 512, 'c_dim': 0, 'w_dim': 512, 'img_resolution': 256, 'img_channels': 3, 'mapping_kwargs': {'num_layers': 8, 'embed_features': None, 'layer_features': None, 'activation': 'lrelu', 'lr_multiplier': 0.01, 'w_avg_beta': 0.995}, 'synthesis_kwargs': {'channel_base': 16384, 'channel_max': 512, 'num_fp16_res': 0, 'conv_clamp': None, 'architecture': 'skip', 'resample_filter': [1, 3, 3, 1], 'use_noise': True, 'activation': 'lrelu'}}
#discriminator arguments
Discriminator_Kwargs= {'c_dim': 0, 'img_resolution': 256, 'img_channels': 3, 'architecture': 'resnet', 'channel_base': 16384, 'channel_max': 512, 'num_fp16_res': 0, 'conv_clamp': None, 'cmap_dim': None, 'block_kwargs': {'activation': 'lrelu', 'resample_filter': [1, 3, 3, 1], 'freeze_layers': 0}, 'mapping_kwargs': {'num_layers': 0, 'embed_features': None, 'layer_features': None, 'activation': 'lrelu', 'lr_multiplier': 0.1}, 'epilogue_kwargs': {'mbstd_group_size': None, 'mbstd_num_channels': 1, 'activation': 'lrelu'}}


#loading of the generator
generator = networks.Generator(Generator_Kwargs['z_dim'], Generator_Kwargs['c_dim'], Generator_Kwargs['w_dim'], Generator_Kwargs['img_resolution'], Generator_Kwargs['img_channels'], Generator_Kwargs['mapping_kwargs'], Generator_Kwargs['synthesis_kwargs'])
generator.load_state_dict(data['G'].state_dict())
generator.to(device)

#loading of the discriminator
discriminator = networks.Discriminator(Discriminator_Kwargs['c_dim'], Discriminator_Kwargs['img_resolution'], Discriminator_Kwargs['img_channels'], Discriminator_Kwargs['architecture'], Discriminator_Kwargs['channel_base'], Discriminator_Kwargs['channel_max'], Discriminator_Kwargs['num_fp16_res'], Discriminator_Kwargs['conv_clamp'], Discriminator_Kwargs['cmap_dim'], Discriminator_Kwargs['block_kwargs'], Discriminator_Kwargs['mapping_kwargs'], Discriminator_Kwargs['epilogue_kwargs'])
discriminator.load_state_dict(data['D'].state_dict())
discriminator.to(device)

# Set random seed for reproducibility
manualSeed = 999

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
#Sets the seed for generating random numbers
torch.manual_seed(manualSeed)



# Root directory for dataset
dataroot = "/home/users/ichekam/data-efficient-gans/DiffAugment-stylegan2-pytorch/flower_dataset"

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 16

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

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

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create batch of latent vectors (latent vectors are intermediate representations) that we will use to visualize
#  the progression of the generator
fixed_noise = noise = torch.randn(32, generator.z_dim, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

SAVE_PER_TIMES = 250
# WGAN values from paper
learning_rate = 0.0001#1e-4
b1 = 0.5
b2 = 0.999
# Setup Adam optimizers for both G and D
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))

#number of iteration
generator_iters = 50000
critic_iter = 5
lambda_term = 10
 # Apply sigmoid to output **NEW**
#m = nn.Sigmoid()

# Lists to keep track of progress
#img_list = []
#G_losses = []
#D_losses = []
#iters = 0



def calculate_gradient_penalty(real_images, fake_images):
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    if cuda:
        eta = eta.cuda(cuda_index)
    else:
        eta = eta

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    if cuda:
        interpolated = interpolated.cuda(cuda_index)
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated, None)
    #prob_interpolated = m(prob_interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).cuda(cuda_index) if cuda else torch.ones(
                                prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    #alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = alpha.to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)
    d_interpolates = D(interpolates, None)
    d_interpolates = d_interpolates.to(device)
    #fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = fake.to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

data = get_infinite_batches(dataloader)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1

for g_iter in range(generator_iters):
    # Requires grad, Generator requires_grad = False
    for p in discriminator.parameters():
        p.requires_grad = True

    # d_loss_real = 0
    # d_loss_fake = 0
    # Wasserstein_D = 0
    # WGAN - Training discriminator more iterations than generato
    for d_iter in range(critic_iter):
        discriminator.zero_grad()

        images = data.__next__()
        # Check for batch to have full batch_size
        if (images.size()[0] != batch_size):
            continue
        
        #z = torch.rand((batch_size, 100, 1, 1), requires_grad=True).to(device)
        images = images.to(device)
        
        #Batch of real images
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # Training the discriminator
       
        # Train with real images
        d_loss_real = discriminator(images, label)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone, retain_graph=True)
        
        # Train with fake images
        noise = torch.randn(batch_size, generator.z_dim, device=device)
        fake_images = generator(noise, label)
        d_loss_fake = discriminator(fake_images, label)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)

        # Train with gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, images.data, fake_images.data)
        gradient_penalty.backward()

        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        d_optimizer.step()
        print(f'  Discriminator iteration: {d_iter}/{critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

        # Generator update
        #for p in generator.parameters():
        #    p.requires_grad = False  # to avoid computation
        
        generator.zero_grad()
        # train generator
        noise = torch.randn(batch_size, generator.z_dim, device=device)
        fake_images = generator(noise, label)
        # compute loss with fake images
        g_loss = discriminator(fake_images, label)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        
        g_cost = -g_loss
        g_optimizer.step()
        print(f'Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}')
        # Saving model and sampling images every 1000th generator iterations
        if (g_iter) % SAVE_PER_TIMES == 0 and d_iter == 0:
            torch.save(generator.state_dict(), f'final2/generated_model/wgan/generator_{g_iter}.pkl')
            torch.save(discriminator.state_dict(), f'final2/generated_model/wgan/discriminator_{g_iter}.pkl')
            print('Models save to ./generator.pkl & ./discriminator.pkl ')
            # # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
            # # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
            # # This way Inception score is more correct since there are different generated examples from every class of Inception model
            # sample_list = []
            # for i in range(125):
            #     samples  = self.data.__next__()
            # #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
            # #     samples = self.G(z)
            #     sample_list.append(samples.data.cpu().numpy())
            # #
            # # # Flattening list of list into one list
            # new_sample_list = list(chain.from_iterable(sample_list))
            # print("Calculating Inception Score over 8k generated images")
            # # # Feeding list of numpy arrays
            # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
            #                                       resize=True, splits=10)

            if not os.path.exists('training_result_images/'):
                os.makedirs('training_result_images/')

            # Denormalize images and save them in grid 8x8
            #noise = torch.randn(batch_size, generator.z_dim, device=device)
            samples = generator(fixed_noise, label)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = utils.make_grid(samples)
            utils.save_image(grid, 'final2/generated_images/wgan/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

            # Testing
            #time = t.time() - self.t_begin
            #print("Real Inception score: {}".format(inception_score))
            print("Generator iter: {}".format(g_iter))
            #print("Time {}".format(time))

            # Write to file inception_score, gen_iters, time
            #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
            #self.file.write(output)
        #print(len(images), images[0])

