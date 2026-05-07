import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet

BATCH_SIZE = 64
EPOCHS     = 5
LR         = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('../data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

model     = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

print("Starting training...\n")

for epoch in range(EPOCHS):
    # ── Training ──────────────────────────────────────────────────────
    model.train()
    running_loss  = 0.0
    train_correct = 0

    for x, y in train_loader:
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss  += loss.item()
        train_correct += (out.argmax(dim=1) == y).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_acc      = 100 * train_correct / len(train_data)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # ── Validation ────────────────────────────────────────────────────
    model.eval()
    val_loss    = 0.0
    val_correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            out  = model(x)
            loss = criterion(out, y)
            val_loss    += loss.item()
            val_correct += (out.argmax(dim=1) == y).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_acc      = 100 * val_correct / len(test_data)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), '../checkpoints/mnist_cnn.pth')
print("\nDone! Model saved.")

# ── Plots ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, marker='o', color='blue',  label='Train')
ax1.plot(range(1, EPOCHS+1), val_losses,   marker='o', color='red',   label='Validation')
ax1.set_title('Train vs Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(range(1, EPOCHS+1), train_accuracies, marker='o', color='blue',  label='Train')
ax2.plot(range(1, EPOCHS+1), val_accuracies,   marker='o', color='green', label='Validation')
ax2.set_title('Train vs Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()

plt.tight_layout()
plt.savefig('../results/training_curves.png')
plt.close()
print("Saved training_curves.png")