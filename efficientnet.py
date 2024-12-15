import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import matplotlib.pyplot as plt
import numpy as np

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to data directories
train_dir = 'plants/train'
val_dir = 'plants/valid'
test_dir = 'plants/test'

# Data transformations for training with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Data transformations for validation and testing
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

# DataLoader setup
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Load and modify EfficientNet-B4 model
model = models.efficientnet_b4(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers except classifier

num_classes = 41
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Update classifier layer
model.classifier[1].requires_grad = True  # Enable training for classifier layer

model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
scaler = GradScaler()

def train(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    total_correct = 0

    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_loader.dataset), total_correct / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()

    return total_loss / len(val_loader.dataset), total_correct / len(val_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()

    return total_correct / len(test_loader.dataset)

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def class_wise_performance(model, dataloader, device, class_names):
    model.eval()
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    for i in range(len(class_names)):
        if class_total[i] > 0:
            print(f'Accuracy of {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of {class_names[i]}: N/A')

def main():
    num_epochs = 5
    best_val_accuracy = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved!")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    test_accuracy = test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    model.load_state_dict(torch.load('best_model.pth'))
    final_test_accuracy = test(model, test_loader, device)
    print(f"Final Test Accuracy with Best Model: {final_test_accuracy:.4f}")

    class_names = train_dataset.classes
    class_wise_performance(model, test_loader, device, class_names)

if __name__ == "__main__":
    main()
