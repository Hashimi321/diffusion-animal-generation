
import torch
import matplotlib.pyplot as plt
from MSDS25066_05_train import UNet, generate_image, alpha_bars, betas, alphas, T

def load_model(model_path, device="cpu"):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_generated_image(generated_tensor, output_path):
    img = (generated_tensor.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "saved_models/diffusion_model-200epoch.pth"

    print(f"Loading model from {model_path} on {device}...")
    model = load_model(model_path, device=device)

    print("Generating image from pure noise...")
    generated = generate_image(model, alpha_bars, betas, alphas, T, device=device)

    save_generated_image(generated, "Graphs_Visualization/test_output.png")