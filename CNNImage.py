import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Simple Toy CNN Model
class ToyCNN(nn.Module):
    def __init__(self):
        super(ToyCNN, self).__init__()
        # Simple 2-layer CNN
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)    # 1 input channel, 8 output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)   # 8 input channels, 16 output channels
        self.fc1 = nn.Linear(16 * 5 * 5, 32)           # Flattened conv output to 32 neurons
        self.fc2 = nn.Linear(32, 10)                    # 32 neurons to 10 classes
        
    def forward(self, x):
        # Conv layers with ReLU and MaxPool
        x = F.relu(self.conv1(x))          # 28x28 -> 26x26
        x = F.max_pool2d(x, 2)             # 26x26 -> 13x13
        x = F.relu(self.conv2(x))          # 13x13 -> 11x11
        x = F.max_pool2d(x, 2)             # 11x11 -> 5x5
        
        # Flatten and fully connected layers
        x = x.view(-1, 16 * 5 * 5)         # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Even simpler model
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)    # Very simple: 1->4 channels
        self.fc1 = nn.Linear(4 * 12 * 12, 10)          # Direct to output
        
    def forward(self, x):
        x = F.relu(self.conv1(x))          # 28x28 -> 24x24
        x = F.max_pool2d(x, 2)             # 24x24 -> 12x12
        x = x.view(-1, 4 * 12 * 12)        # Flatten
        x = self.fc1(x)
        return x

# Quick data loader for MNIST
def get_toy_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Small batch size for toy model
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Simple training function
def train_toy_model(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.1f}%')

# Test function
def test_model(model, test_loader):
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
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Visualize some predictions
def visualize_predictions(model, test_loader, num_images=6):
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        data, target = next(iter(test_loader))
        output = model(data)
        pred = output.argmax(dim=1)
        
        for i in range(num_images):
            img = data[i].squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {target[i]}, Pred: {pred[i]}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Creating toy CNN model...")
    
    # Choose model (ToyCNN is slightly more complex, MiniCNN is ultra-simple)
    model = ToyCNN()  # or MiniCNN() for even simpler
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load data
    print("Loading MNIST data...")
    train_loader, test_loader = get_toy_data()
    
    # Train
    print("Training model...")
    train_toy_model(model, train_loader, epochs=3)
    
    # Test
    print("Testing model...")
    test_model(model, test_loader)
    
    # Visualize
    print("Visualizing predictions...")
    visualize_predictions(model, test_loader)
    
    # Show model architecture
    print("\nModel Architecture:")
    print(model)

# Quick demo with random data (no download needed)
def demo_with_random_data():
    print("\n=== Quick Demo with Random Data ===")
    
    model = MiniCNN()
    
    # Create random data (batch_size=10, channels=1, height=28, width=28)
    random_input = torch.randn(10, 1, 28, 28)
    random_labels = torch.randint(0, 10, (10,))
    
    # Forward pass
    output = model(random_input)
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample): {output[0]}")
    
    # Quick training step
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    loss = criterion(output, random_labels)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training step completed!")

# Run demo
demo_with_random_data()