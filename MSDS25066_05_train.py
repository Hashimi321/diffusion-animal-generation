
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math

class AnimalDiffusionDataset(Dataset):
    def __init__(self, root_dir, classes, images_per_class=20, transform=None):
        self.transform = transform
        self.image_paths = []
        for class_name in classes:
            class_folder = os.path.join(root_dir, class_name)
            all_files = os.listdir(class_folder)
            original_files = []
            for filename in all_files:
                name_without_ext = filename.rsplit(".", 1)[0]
                parts = name_without_ext.split("_")
                if len(parts) == 2:
                    original_files.append(filename)
            selected_files = original_files[:images_per_class]
            for filename in selected_files:
                full_path = os.path.join(class_folder, filename)
                self.image_paths.append(full_path)       

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
    
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def forward_diffusion(x0, t, alpha_bars):
    # x0: the clean image tensor, shape [3, 64, 64]
    # t: which timestep (an integer, e.g. 500)
    # alpha_bars: the precomputed lookup table we just built
    
    # 1. Look up alpha_bar at this specific timestep t
    alpha_bar_t = alpha_bars[t]
    # 2. Generate random Gaussian noise, same shape as x0
    noise = torch.randn_like(x0)
    # 3. Apply the formula:  xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise 
    # 4. Return both xt and the noise itself (you'll need the noise later for the loss function!)
    return xt, noise

def custom_loss(predicted_noise, actual_noise):
    difference = predicted_noise - actual_noise
    squared_difference = difference ** 2
    loss = squared_difference.mean()
    return loss


@torch.no_grad()
def generate_image(model, alpha_bars, betas, alphas, T, device="cpu", image_size=64):
    model.eval()
    x = torch.randn((1, 3, image_size, image_size)).to(device)

    for t_step in reversed(range(T)):
        t = torch.tensor([t_step]).to(device)
        predicted_noise = model(x, t)

        alpha_t = alphas[t_step].to(device)
        alpha_bar_t = alpha_bars[t_step].to(device)
        beta_t = betas[t_step].to(device)

        if t_step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

    model.train()
    return x

def train_model(model, dataset, epochs=5, batch_size=4, learning_rate=1e-4, device="cpu"):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            batch_size_actual = batch.shape[0]
            t = torch.randint(0, T, (batch_size_actual,)).to(device)

            xt_list = []
            noise_list = []
            for i in range(batch_size_actual):
                xt, noise = forward_diffusion(batch[i], t[i], alpha_bars.to(device))
                xt_list.append(xt)
                noise_list.append(noise)
            xt_batch = torch.stack(xt_list)
            noise_batch = torch.stack(noise_list)

            predicted_noise = model(xt_batch, t)
            loss = custom_loss(predicted_noise, noise_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "saved_models/diffusion_model.pth")
    print("Model saved to saved_models/diffusion_model.pth")
    return loss_history

class TimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
         half_dim = self.embedding_dim // 2
         freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
         args = t[:, None].float() * freqs[None, :]
         embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
         return embedding


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(8, out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(8, out_channels)
        self.activation = torch.nn.SiLU()
        self.time_proj = torch.nn.Linear(time_embedding_dim, out_channels)

    def forward(self, x, t_embed):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        t = self.time_proj(t_embed)
        t = t[:, :, None, None]
        x = x + t

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, time_embedding_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        self.time_proj = torch.nn.Linear(time_embedding_dim, time_embedding_dim)

        # Down path
        self.down1 = ConvBlock(3, 64, time_embedding_dim)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.down2 = ConvBlock(64, 128, time_embedding_dim)
        self.pool2 = torch.nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256, time_embedding_dim)

        # Up path
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ConvBlock(256 + 128, 128, time_embedding_dim)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = ConvBlock(128 + 64, 64, time_embedding_dim)

        # Output
        self.output_conv = torch.nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, t):
        # Step A: turn timestep into a vector
        t_embed = self.time_embedding(t)
        t_embed = self.time_proj(t_embed)

        # Step B: down path — shrink the image, remember each stage (skip connections)
        skip1 = self.down1(x, t_embed)
        x = self.pool1(skip1)

        skip2 = self.down2(x, t_embed)
        x = self.pool2(skip2)

        # Step C: bottleneck — smallest, most compressed point
        x = self.bottleneck(x, t_embed)

        # Step D: up path — grow the image back, reusing the skip connections
        x = self.upsample1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up1(x, t_embed)

        x = self.upsample2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up2(x, t_embed)

        # Step E: final output — predicted noise, same shape as input image
        x = self.output_conv(x)
        return x
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a diffusion model on animal images")
    parser.add_argument("--data_path", type=str, default="animal_data", help="Path to the dataset folder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--images_per_class", type=int, default=20, help="Number of images to use per class")
    args = parser.parse_args()

    classes = ["Bear", "Cat", "Dog", "Lion", "Tiger"]
    dataset = AnimalDiffusionDataset(root_dir=args.data_path, classes=classes, images_per_class=args.images_per_class, transform=image_transform)
    print("Total images found:", len(dataset.image_paths))
    print("First 3 paths:", dataset.image_paths[:3])
    sample_image = dataset[0]
    print("Sample tensor shape:", sample_image.shape)
    print("Min value:", sample_image.min().item())
    print("Max value:", sample_image.max().item())

    test_timesteps = [0, 100, 300, 500, 700, 999]
    fig, axes = plt.subplots(1, len(test_timesteps), figsize=(15, 3))

    for i, t in enumerate(test_timesteps):
        xt, noise = forward_diffusion(sample_image, t, alpha_bars)
        img_to_show = (xt.clamp(-1, 1) + 1) / 2
        img_to_show = img_to_show.permute(1, 2, 0).numpy()
        axes[i].imshow(img_to_show)
        axes[i].set_title(f"t={t}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("forward_diffusion_test.png")
    print("Saved forward_diffusion_test.png — open it to check the progression")

    model = UNet()
    test_batch = sample_image.unsqueeze(0)
    test_t = torch.tensor([500])
    predicted_noise = model(test_batch, test_t)
    print("Input shape:", test_batch.shape)
    print("Predicted noise shape:", predicted_noise.shape)

    print("\nStarting training...")
    train_dataset = AnimalDiffusionDataset(root_dir=args.data_path, classes=classes, images_per_class=args.images_per_class, transform=image_transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    train_model(model, train_dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device=device)
