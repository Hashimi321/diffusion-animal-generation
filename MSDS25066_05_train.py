
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


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



if __name__ == "__main__":
    classes = ["Bear", "Cat", "Dog", "Lion", "Tiger"]
    dataset = AnimalDiffusionDataset(root_dir="animal_data", classes=classes, transform=image_transform)
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