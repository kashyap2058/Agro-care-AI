import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directory structure
data_dir = 'dataset/'
train_dir = data_dir + 'train'
val_dir = data_dir + 'val'
test_img_dir = data_dir + 'test/test'

# Image Preprocessing
image_size = 256
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load train and validation datasets
batch_size = 32
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Class names
class_names = train_dataset.classes
num_classes = 41

# Custom Dataset class to load test images
# class TestDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, img_path

# Load the test dataset
# test_dataset = TestDataset(test_img_dir, transform=data_transforms['val'])
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Load pre-trained ResNet101 and modify the fully connected layer
model = models.resnet101(weights=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scaler = GradScaler()

# Gradient Accumulation settings
accumulation_steps = 4

# Training and Validation Loop with Mixed Precision and Gradient Accumulation
def train_model(model, criterion, optimizer, num_epochs=12):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f'{phase.capitalize()} Epoch {epoch+1}')):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

# Train the model
train_loss, val_loss, train_acc, val_acc = train_model(model, criterion, optimizer, num_epochs=12)

# Plot Loss and Accuracy Graphs
def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.plot(epochs, val_loss, 'b-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the performance metrics
plot_metrics(train_loss, val_loss, train_acc, val_acc)

print("Training complete!")

# Testing the model is done separately
# Testing the model
# def test_model_classwise(model, class_names):
#     model.eval()
#     all_preds = []
#     image_paths = []

#     with torch.no_grad():
#         for inputs, paths in tqdm(test_loader, desc='Testing'):
#             inputs = inputs.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             all_preds.extend(preds.cpu().numpy())
#             image_paths.extend(paths)

#     for i, path in enumerate(image_paths):
#         print(f"Image: {path}, Predicted class: {class_names[all_preds[i]]}")

# # Evaluate on test set with predictions
# test_model_classwise(model, class_names)

# Save the trained model
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
