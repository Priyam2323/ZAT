import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


import torch.nn as nn
from torchvision.utils import save_image


def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    return magnitude_spectrum



def train_encoder_izif(opt, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("run2/generator_gdea_1.5"))
    discriminator.load_state_dict(torch.load("run2/discriminator_gdea_1.5"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("run2/images_tb", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            real_features = discriminator.forward_features(real_imgs)
            # Fake features
            fake_features = discriminator.forward_features(fake_imgs)

            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            fake_img_cpu=fake_imgs    
            real_img_cpu = real_imgs
            fake_img_cpu = fake_img_cpu.detach().cpu().numpy()
            real_img_cpu = real_img_cpu.detach().cpu().numpy()

            fft_image1 = compute_fft(real_img_cpu)
            fft_image2 = compute_fft(fake_img_cpu)

            # Normalize the magnitude spectra for comparison
            fft_image1_norm = fft_image1 / np.max(fft_image1)
            fft_image2_norm = fft_image2 / np.max(fft_image2)
            #similarity=criterion(fft_image1_norm,fft_image2_norm)
            # Compute the difference
            diff = np.abs(fft_image1_norm - fft_image2_norm)

            # Compute the mean difference as a similarity measure
            similarity = np.mean(diff)
            similarity=torch.tensor(similarity).to(device)

        
            #e_loss = (0.5*loss_imgs) + (0.25 * loss_features)+(0.25*similarity)
            e_loss = (loss_imgs) + (loss_features)
            e_loss.backward()
            optimizer_E.step()

            # Output training log every n_critic steps
            if i % opt.n_critic == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

                if batches_done % opt.sample_interval == 0:
                    fake_z = encoder(fake_imgs)
                    reconfiguration_imgs = generator(fake_z)
                    save_image(reconfiguration_imgs.data[:25],
                               f"run2/images_tb/{batches_done:06}.png",
                               nrow=5, normalize=True)

                batches_done += opt.n_critic
                
                

	    	
    torch.save(encoder.state_dict(), "run2/encoder_gdea_1.5")



def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipeline = [transforms.Resize([opt.img_size]*2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

    transform = transforms.Compose(pipeline)
    dataset = ImageFolder(opt.train_root, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                                  shuffle=True,num_workers=4)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model_gdea import Generator, Discriminator, Encoder

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder_izif(opt, generator, discriminator, encoder,
                       train_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=str,
                        help="root name of your dataset in train mode")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of attention layer")
    opt = parser.parse_args()

    main(opt)
