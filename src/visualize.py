import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import MNISTNet

model = MNISTNet()
model.load_state_dict(torch.load('../checkpoints/mnist_cnn.pth', weights_only=True))
model.eval()

# ── 1. Learned kernels from conv1 ────────────────────────────────────
kernels = model.conv1.weight.detach()  # shape: [32, 1, 3, 3]
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(kernels[i][0], cmap='gray')
    ax.axis('off')
plt.suptitle('Conv1 Learned Kernels (32 filters)', fontsize=12)
plt.tight_layout()
plt.savefig('../results/kernels.png')
plt.close()
print("Saved kernels.png")

# ── 2. Feature maps for one real image ───────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)
img, label = test_data[0]
img_input = img.unsqueeze(0)  # add batch dim -> [1, 1, 28, 28]

with torch.no_grad():
    feature_maps = torch.relu(model.conv1(img_input))  # [1, 32, 28, 28]

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[0][i], cmap='viridis')
    ax.axis('off')
plt.suptitle(f'Conv1 Feature Maps for digit: {label}', fontsize=12)
plt.tight_layout()
plt.savefig('../results/feature_maps.png')
plt.close()
print("Saved feature_maps.png")