import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils


def show_grid(image):
    npimage = image.numpy()
    plt.imshow(np.transpose(npimage, (1, 2, 0)))
    plt.axis("off")
    plt.show()


def plot_output(generator, noise, caption_embed, epoch, save_dir="./output_images"):
    os.makedirs(save_dir, exist_ok=True)
    plt.clf()
    with torch.no_grad():
        generator.eval()
        test_images = generator(noise.to(generator.device), caption_embed.to(generator.device))
        generator.train()
        grid = torchvision.utils.make_grid(test_images.cpu(), normalize=True)
        # Lưu ảnh ra file
        file_path = os.path.join(save_dir, f"epoch_{epoch}.png")
        torchvision.utils.save_image(test_images.cpu(), file_path, nrow=8, normalize=True)
        print(f"Saved generated image at: {file_path}")
