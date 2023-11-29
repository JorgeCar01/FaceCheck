import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

print(torch.cuda.is_available())
# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Loading Dataset")
# Load dataset
path_to_data = r'C:\School\csci 4353\studentData'
dataset = datasets.ImageFolder(root=path_to_data, transform=transform)

model = models.resnet18(pretrained=False, progress=True)
pretrained_dict = torch.load(r'C:\School\csci 4353\FaceCheck\face_recognition_model.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct_preds = 0

    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        correct_preds += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = (correct_preds / len(data_loader.dataset)) * 100

    return avg_loss, accuracy

num_epochs = 10 # Adjust the number of epochs

for epoch in range(num_epochs):
    loss, acc = train_model(model, data_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Save only the state dictionary
torch.save(model.state_dict(), 'facecheck2.pth')
