import os
import time

import torch

from encode_captions import encode_captions
from load_captions import load_captions
from model import Discriminator
from model import Generator
from preprocessing import FlowerDataset
from transform import transform
from utils import plot_output

# Build path từ root
IMAGE_DIR = "./content/cvpr2016_flowers/images"
CAPTION_DIR = "./content/cvpr2016_flowers/captions"

captions = load_captions(IMAGE_DIR, CAPTION_DIR)

encoded_caption = encode_captions(captions)

dataset = FlowerDataset(IMAGE_DIR, captions=encoded_caption, transform=transform)

batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(noise_size=100, feature_size=128, num_channels=3, embedding_size=768, reduced_dim_size=256).to(
    device)
discriminator = Discriminator(3, 128, 768, 256).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

bce_loss = torch.nn.BCELoss()
l2_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()

fixed_noise = torch.randn(5, 100).to(device)

sample_indices = ["image_00001", "image_00002", "image_00003", "image_00004", "image_00005"]
fixed_embed_captions_list = [torch.Tensor(encoded_caption[i]["embed"]) for i in sample_indices]
fixed_embed_captions = torch.stack(fixed_embed_captions_list).to(device)

epochs = 500

g_loss_history = []
d_loss_history = []
for epoch in range(epochs):
    d_losses, g_losses = [], []
    epoch_time = time.time()

    for batch in dataloader:
        images = batch["image"].to(device)
        embed_captions = batch["embed"].to(device)
        wrong_images = batch["wrong_image"].to(device)

        # labels
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # train discriminator
        optimizer_D.zero_grad()

        # gen fake images
        noise = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(noise, embed_captions)

        # compute real loss
        outputs, _ = discriminator(images, embed_captions)
        real_loss = bce_loss(outputs, real_labels)

        # compute contrastive loss for wrong images
        outputs, _ = discriminator(wrong_images, embed_captions)
        wrong_loss = bce_loss(outputs, fake_labels)

        # compute fake loss
        outputs, _ = discriminator(fake_images, embed_captions)
        fake_loss = bce_loss(outputs, fake_labels)

        d_loss = real_loss + wrong_loss + fake_loss

        # update weights
        d_loss.backward()
        optimizer_D.step()
        d_losses.append(d_loss.item())

        # train generator
        optimizer_G.zero_grad()

        noise = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(noise, embed_captions)

        outputs, fake_features = discriminator(fake_images, embed_captions)
        _, real_features = discriminator(images, embed_captions)

        activation_fake = torch.mean(fake_features, dim=0)
        activation_real = torch.mean(real_features, dim=0)

        real_loss = bce_loss(outputs, real_labels)
        g_loss = real_loss + 100 * l2_loss(activation_fake, activation_real) + 50 * l1_loss(fake_images, images)

        g_loss.backward()
        optimizer_G.step()
        g_losses.append(g_loss.item())

    avg_d_loss = sum(d_losses) / len(d_losses)
    avg_g_loss = sum(g_losses) / len(g_losses)
    g_loss_history.append(avg_g_loss)
    d_loss_history.append(avg_d_loss)

    if (epoch + 1) % 10 == 0:
        plot_output(generator, fixed_noise, fixed_embed_captions, epoch=epoch + 1)
    print(
        f"Epoch {epoch + 1}/{epochs}, D Loss: {avg_d_loss}, G Loss: {avg_g_loss}, time taken: {time.time() - epoch_time}")

model_save_path = "./models"
torch.save(generator.state_dict(), os.path.join(model_save_path, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(model_save_path, "discriminator.pth"))
import matplotlib.pyplot as plt

# ==========================
# Vẽ và lưu đồ thị loss
# ==========================
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_loss_history, label="G Loss")
plt.plot(d_loss_history, label="D Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

loss_plot_path = "./output_images/loss_curve.png"
os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
plt.savefig(loss_plot_path)
plt.show()

print(f"Saved loss curve at: {loss_plot_path}")
