import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import MNISTNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data   = datasets.MNIST('../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = MNISTNet()
model.load_state_dict(torch.load('../checkpoints/mnist_cnn.pth', weights_only=True))
model.eval()

all_preds  = []
all_labels = []
all_images = []

with torch.no_grad():
    for x, y in test_loader:
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())
        all_images.extend(x.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_images = np.array(all_images)

# ── 1. Confusion matrix ──────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('../results/confusion_matrix.png')
plt.close()
print("Saved confusion_matrix.png")

# ── 2. Sample predictions grid ───────────────────────────────────────
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    img = all_images[i][0]
    ax.imshow(img, cmap='gray')
    color = 'green' if all_preds[i] == all_labels[i] else 'red'
    ax.set_title(f'P:{all_preds[i]} A:{all_labels[i]}', color=color, fontsize=8)
    ax.axis('off')
plt.suptitle('Sample Predictions (green=correct, red=wrong)', fontsize=12)
plt.tight_layout()
plt.savefig('../results/sample_predictions.png')
plt.close()
print("Saved sample_predictions.png")

# ── 3. Misclassified examples ────────────────────────────────────────
wrong_idx = np.where(all_preds != all_labels)[0]
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for i, ax in enumerate(axes.flat):
    if i >= len(wrong_idx):
        ax.axis('off')
        continue
    idx = wrong_idx[i]
    ax.imshow(all_images[idx][0], cmap='gray')
    ax.set_title(f'P:{all_preds[idx]} A:{all_labels[idx]}', color='red', fontsize=8)
    ax.axis('off')
plt.suptitle('Misclassified Examples', fontsize=12)
plt.tight_layout()
plt.savefig('../results/misclassified.png')
plt.close()
print("Saved misclassified.png")

accuracy = (all_preds == all_labels).mean() * 100
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")