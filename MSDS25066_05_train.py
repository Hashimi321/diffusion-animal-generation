
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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


if __name__ == "__main__":
    classes = ["Bear", "Cat", "Dog", "Lion", "Tiger"]
    dataset = AnimalDiffusionDataset(root_dir="animal_data", classes=classes, transform=image_transform)
    print("Total images found:", len(dataset.image_paths))
    print("First 3 paths:", dataset.image_paths[:3])
    sample_image = dataset[0]
    print("Sample tensor shape:", sample_image.shape)
    print("Min value:", sample_image.min().item())
    print("Max value:", sample_image.max().item())

