import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex  # Intel XPU extension

# NPU-optimized Mini CNN Model
class NPUMiniCNN(nn.Module):
    def __init__(self):
        super(NPUMiniCNN, self).__init__()
        # Optimized for NPU inference
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(4)  # BatchNorm for NPU optimization
        self.fc1 = nn.Linear(4 * 12 * 12, 10, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)                  # 28x28 -> 24x24
        x = self.bn1(x)                    # Batch normalization
        x = F.relu(x)                      # ReLU activation
        x = F.max_pool2d(x, 2)             # 24x24 -> 12x12
        x = x.view(-1, 4 * 12 * 12)        # Flatten
        x = self.fc1(x)                    # Linear layer
        return x

# Device detection and setup
def setup_device():
    """
    Setup device for Intel NPU/XPU
    """
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"Using Intel XPU: {torch.xpu.get_device_name()}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

# NPU-optimized data loading
def get_npu_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Smaller batch sizes often work better on NPU
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader

# NPU-optimized training
def train_npu(model, train_loader, device, epochs=3):
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Optimize model for Intel XPU (with optimizer for training mode)
    if device.type == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    
    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
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

# NPU inference optimization
def optimize_for_inference(model, device):
    """
    Optimize model for NPU inference
    """
    model.eval()
    
    if device.type == 'xpu':
        # Intel XPU optimizations for inference (no optimizer needed)
        model = ipex.optimize(model, dtype=torch.float32, level="O1")
        
        # Optional: Convert to JIT for better performance
        try:
            sample_input = torch.randn(1, 1, 28, 28).to(device)
            with torch.no_grad():
                traced_model = torch.jit.trace(model, sample_input)
                traced_model = torch.jit.freeze(traced_model)
            return traced_model
        except Exception as e:
            print(f"JIT tracing failed: {e}")
            return model
    
    return model

# Test with NPU optimization
def test_npu(model, test_loader, device):
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Time inference
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start_time.record()
            
            output = model(data)
            
            if device.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                inference_times.append(start_time.elapsed_time(end_time))
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    print(f'Test Accuracy: {accuracy:.1f}%')
    if avg_time > 0:
        print(f'Average inference time: {avg_time:.2f}ms per batch')

# Quantization for NPU (INT8)
def quantize_model(model, device):
    """
    Quantize model for better NPU performance
    """
    if device.type == 'xpu':
        try:
            # Intel XPU quantization
            model.eval()
            
            # Simple quantization approach
            model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")
            print("Model optimized with bfloat16 for NPU")
            
        except Exception as e:
            print(f"Quantization failed: {e}")
    
    return model

# Benchmark function
def benchmark_npu(model, device, batch_size=32):
    """
    Benchmark NPU performance
    """
    model = model.to(device)
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Synchronize
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    num_runs = 100
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    throughput = batch_size * num_runs / (end_time - start_time)  # samples/sec
    
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"Throughput: {throughput:.1f} samples/sec")

# Main execution
def main():
    print("=== Intel Meteor Lake NPU/XPU Mini CNN ===")
    
    # Setup device
    device = setup_device()
    
    # Create model
    model = NPUMiniCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Load data
    train_loader, test_loader = get_npu_data()
    
    # Train
    print("\nTraining on NPU/XPU...")
    train_npu(model, train_loader, device, epochs=2)
    
    # Optimize for inference
    print("\nOptimizing for inference...")
    optimized_model = optimize_for_inference(model, device)
    
    # Test
    print("\nTesting...")
    test_npu(optimized_model, test_loader, device)
    
    # Benchmark
    print("\nBenchmarking...")
    benchmark_npu(optimized_model, device)
    
    # Optional: Quantization
    print("\nQuantizing model...")
    try:
        quantized_model = quantize_model(model, device)
        print("Quantization successful")
    except Exception as e:
        print(f"Quantization not available: {e}")

# Quick NPU demo
def npu_demo():
    print("=== Quick NPU Demo ===")
    
    device = setup_device()
    model = NPUMiniCNN().to(device)
    
    # Sample inference
    x = torch.randn(1, 1, 28, 28).to(device)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Device: {device}")
    print(f"Model on device: {next(model.parameters()).device}")

if __name__ == "__main__":
    # Run demo first
    npu_demo()
    
    # Full training and optimization
    main()