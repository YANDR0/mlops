import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

IMG_SIZE = 128
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 3
EPOCHS = 50

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder("data/train", transform=transform_train)
test_data  = datasets.ImageFolder("data/test",  transform=transform_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print("Data loaded successfully.")
print("Classes:", train_data.classes)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(train_data.classes))
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=1e-4)

best_acc = 0
wait = 0

print("Training started...")

for epoch in range(EPOCHS):

    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Val Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")

