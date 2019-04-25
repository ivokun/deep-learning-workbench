import argparse
import os
import numpy as np
import math

from dataloader import load_gan_data
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
# CUDA_LAUNCH_BLOCKING=1
os.makedirs("new_dataset", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
  "--n_epochs", type=int, default=10, help="number of epochs of training"
  )
parser.add_argument(
  "--batch_size", type=int, default=10, help="size of the batches"
  )
parser.add_argument(
  "--lr", type=float, default=0.0002, help="adam: learning rate"
  )
parser.add_argument(
  "--b1", 
  type=float, 
  default=0.5, 
  help="adam: decay of first order momentum of gradient"
  )
parser.add_argument(
  "--b2", 
  type=float, 
  default=0.999, 
  help="adam: decay of first order momentum of gradient"
  )
parser.add_argument(
  "--n_cpu", 
  type=int, 
  default=8,
  help="number of cpu threads to use during batch generation"
  )
parser.add_argument(
  "--latent_dim",
  type=int, 
  default=100, 
  help="dimensionality of the latent space"
  )
parser.add_argument(
  "--n_classes", 
  type=int, 
  default=1, 
  help="number of classes for dataset"
  )
parser.add_argument(
  "--input_size", 
  type=int, 
  default=32, 
  help="size of each image dimension"
  )
parser.add_argument(
  "--channels", 
  type=int, 
  default=1, 
  help="number of image channels"
  )
parser.add_argument(
  '--dataset',
  type=str,
  default='vote',
  help='The name of dataset'
  )
parser.add_argument(
  "--sample_interval", 
  type=int, 
  default=10, 
  help="interval between image sampling"
  )

opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False
# cuda = False

# Configure data loader
dataloader, features_num, gan_size = load_gan_data(opt.dataset, opt.batch_size)

input_shape = features_num

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.batch_size, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, features_num),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *input_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.batch_size, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )

    def forward(self, img, labels):
        # print("INPUT")
        # print(img)
        # print("LABELS")
        # print(labels)
        # print("LABEL EMBEDDING")
        # print(self.label_embedding(labels))
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        # print("D_IN")
        # print(d_in)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_data(dataset, batch_size, batches_done):
  z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
  # Get labels ranging from 0 to n_classes for n rows
  labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
  gen_data = generator(z, labels)
  datas = gen_data.tolist()
  # print(gen_data.tolist())
  with open(dataset + ".txt", "a") as f:
    for data in datas:
      f.write("%s\n" % data)
       

# ----------
#  Training
# ----------
print("-"*15)
print("Training Start")
print("-"*15)
for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):

        imgs = data['features']
        labels = data['target']
        batch_size = imgs.shape[0]
        # print("label shape")
        # print(labels.shape)
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        labels = labels.reshape([labels.size(0)])

        # -----------------
        #  Train Generator
        # -----------------

        # print("Train Generator")
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        # print(gen_imgs.shape)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # print("Train Discriminator")
        optimizer_D.zero_grad()
        validity_real = discriminator(real_imgs, labels)
        
        d_real_loss = adversarial_loss(validity_real, valid)


        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # sample_data(opt.dataset, n_row=10, batches_done=batches_done)
            print("SAVE BATCHES")
            sample_data(opt.dataset, batch_size=batch_size, batches_done=batches_done)