import torch
import matplotlib.pyplot as plt
from models.cvae import ConditionalVAE  # adjust if path differs

# -----------------------------
# Config
# -----------------------------
B = 2                # batch size
C = 1                # grayscale
H = W = 64           # image size
latent_dim = 128
num_classes = 15

# -----------------------------
# Fake Data
# -----------------------------
# Simulated grayscale X-rays
x = torch.randn(B, C, H, W)

# Multi-hot disease labels
y = torch.zeros(B, num_classes)
for i in range(B):
    active_classes = torch.randint(0, num_classes, (3,))
    y[i, active_classes] = 1.0

print(f"y: {y}")
print(f"x: {x}")
# -----------------------------
# Model
# -----------------------------
model = ConditionalVAE(in_channels=C, latent_dim=latent_dim, num_classes=num_classes)

# -----------------------------
# Forward Pass
# -----------------------------
x_recon, x_input, mu, log_var = model(x, y)

print("x shape:         ", x.shape)
print("y shape:         ", y.shape)
print("x_recon shape:   ", x_recon.shape)
print("mu shape:        ", mu.shape)
print("log_var shape:   ", log_var.shape)

# -----------------------------
# Conditioning Check
# -----------------------------
with torch.no_grad():
    x_fixed = x[0].unsqueeze(0)  # [1, 1, 64, 64]
    
    y_original = y[0].unsqueeze(0)
    y_zero = torch.zeros_like(y_original)
    y_one = torch.ones_like(y_original)

    out_original = model(x_fixed, y_original)[0]
    out_zero = model(x_fixed, y_zero)[0]
    out_one = model(x_fixed, y_one)[0]

    diff_1 = torch.mean((out_original - out_zero) ** 2).item()
    diff_2 = torch.mean((out_original - out_one) ** 2).item()

    print(f"\nChange in output from altering y:")
    print(f"  MSE (original vs zeros): {diff_1:.6f}")
    print(f"  MSE (original vs ones):  {diff_2:.6f}")

# -----------------------------
# Visualization
# -----------------------------
# De-normalize (from Tanh [-1, 1] → [0, 1])
def denorm(img): return (img + 1) / 2

images = [out_original, out_zero, out_one]
titles = ["Original label", "All Zeros", "All Ones"]

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for i in range(3):
    axes[i].imshow(denorm(images[i][0, 0]).cpu(), cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')
plt.tight_layout()
plt.show()
