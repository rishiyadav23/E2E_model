import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ultra-simple Mini CNN Model
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)    # 1->4 channels, 5x5 kernel
        self.fc1 = nn.Linear(4 * 12 * 12, 10)          # Direct to 10 classes
        
    def forward(self, x):
        x = F.relu(self.conv1(x))          # 28x28 -> 24x24
        x = F.max_pool2d(x, 2)             # 24x24 -> 12x12
        x = x.view(-1, 4 * 12 * 12)        # Flatten: 4*12*12 = 576
        x = self.fc1(x)                    # 576 -> 10
        return x

# Simple data loader
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.3f}, Acc: {100.*correct/total:.1f}%')

# Test function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.1f}%')

# Quick demo with random data
def demo():
    print("=== Mini CNN Demo ===")
    
    model = MiniCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print("Model:", model)
    
    # Random data demo
    x = torch.randn(1, 1, 28, 28)  # Single image
    y = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y[0]}")

# Main execution
if __name__ == "__main__":
    # Quick demo first
    demo()
    
    # Full training
    print("\n=== Training on MNIST ===")
    model = MiniCNN()
    
    train_loader, test_loader = get_data()
    
    print("Training...")
    train(model, train_loader, epochs=2)
    
    print("Testing...")
    test(model, test_loader)