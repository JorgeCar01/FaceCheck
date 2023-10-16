import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 1. Data Loading and Preprocessing:

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Digi-Face 1M
digi_face_dataset = datasets.ImageFolder(root='path_to_digi_face', transform=transform)

# Load CelebA
celeba_dataset = datasets.ImageFolder(root='path_to_celeba', transform=transform)

# 2. Model Definition:
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(digi_face_dataset.classes))  # Adjust for number of classes in Digi-Face 1M

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training:

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training on Digi-Face 1M first:

train_size = int(0.8 * len(digi_face_dataset))
val_size = len(digi_face_dataset) - train_size
train_dataset, val_dataset = random_split(digi_face_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer)
    val_loss = evaluate_model(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} (Digi-Face 1M) - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Fine-tuning on CelebA:

model.fc = nn.Linear(model.fc.in_features, len(celeba_dataset.classes))  # Adjust for number of classes in CelebA

train_size = int(0.8 * len(celeba_dataset))
val_size = len(celeba_dataset) - train_size
train_dataset, val_dataset = random_split(celeba_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer)
    val_loss = evaluate_model(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} (CelebA) - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "face_recognition_model_finetuned.pth")
