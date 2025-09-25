import os
import sys
import torchvision.utils
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

#from fanogan.test_anomaly_detection import test_anomaly_detection

from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm
import matplotlib.pyplot as plt

def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    return magnitude_spectrum



def test_anomaly_detection(opt, generator, discriminator, encoder,dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("run2/generator_gdea_1.5"))
    discriminator.load_state_dict(torch.load("run2/discriminator_gdea_1.5"))
    encoder.load_state_dict(torch.load("run2/encoder_gdea_1.5"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss()

    with open("run2/score_1.5.csv", "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance,similarity\n")
    i=1
    for (img, label) in tqdm(dataloader):

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature
        residual_map = torch.abs(real_img - fake_img)
        #torchvision.utils.save_image(residual_map, f'Residual_Map{i}.png')
        #i=i+1
        #print("Residual image saved successfully!")
        z_distance = criterion(fake_z, real_z)

        fake_img_cpu = fake_img.detach().cpu().numpy()
        real_img_cpu = real_img.detach().cpu().numpy()

        fft_image1 = compute_fft(real_img_cpu)
        fft_image2 = compute_fft(fake_img_cpu)

        # Normalize the magnitude spectra for comparison
        fft_image1_norm = fft_image1 / np.max(fft_image1)
        fft_image2_norm = fft_image2 / np.max(fft_image2)

        # Compute the difference
        diff = np.abs(fft_image1_norm - fft_image2_norm)
       

        # Compute the mean difference as a similarity measure
        similarity = np.mean(diff)


        with open("run2/score_1.5.csv", "a") as f:
            f.write(f"{label.item()},{img_distance},"
                    f"{anomaly_score},{z_distance},{similarity.item()}\n")



def main(opt):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    pipeline = [transforms.Resize([opt.img_size]*2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

    transform = transforms.Compose(pipeline)
    dataset = ImageFolder(opt.test_root, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model_gdea import Generator, Discriminator, Encoder


    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=str,
                        help="root name of your dataset in test mode")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    opt = parser.parse_args()

    main(opt)
