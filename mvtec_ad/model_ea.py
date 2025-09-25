import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def init_weights(m):
	if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
		torch.nn.init.normal_(m.weight,0.0,0.02)
	elif isinstance(m,nn.BatchNorm2d):
		torch.nn.init.normal_(m.weight,1.0,0.02)
		torch.nn.init.constant_(m.bias,0)
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,128 * self.init_size ** 2))
        #self.attention = SelfAttention(128)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
	        nn.ConvTranspose2d(128,128,4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Apply self-attention layer here
            #self.attention,
            
	        nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            )
        self.apply(init_weights)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
        
        
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            #SelfAttention(16),  # Adding self-attention layer
            *discriminator_block(16, 32),
            #SelfAttention(32),  # Adding self-attention layer
            *discriminator_block(32, 64),
            #SelfAttention(64),  # Adding self-attention layer
            *discriminator_block(64, 128),
            )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        
        self.apply(init_weights)

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        return features


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder,self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *encoder_block(opt.channels, 16, bn=False),
            #SelfAttention(16),  # Adding self-attention layer
            *encoder_block(16, 32),
            SelfAttention(32),  # Adding self-attention layer
            *encoder_block(32, 64),
            SelfAttention(64),  # Adding self-attention layer
            *encoder_block(64, 128),
            )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.latent_dim),nn.Tanh())                             
        self.apply(init_weights)
        
        
        
    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity
