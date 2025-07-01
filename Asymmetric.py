import numpy as np
import torch
import matplotlib.pyplot as plt

def asymmetric_quantize(tensor, dtype=torch.int8):
	q_min = torch.iinfo(dtype).min
	q_max = torch.iinfo(dtype).max

	r_max = tensor.max().item()
	r_min = tensor.min().item()

	# Calculate scale using the below formula
	scale = (r_max - r_min)/ (q_max - q_min)

	# Calculate zero point
	zero = round(q_min - (r_min/scale))

	# Quantization
	quantized_tensor = torch.round((tensor - zero_point) / scale).to(torh.int8)

	# Clamp to ensure values are within range
	quantized_tensor = torch.clamp(quantized_tensor, q_min, q_max)

	return quantized_tensor, scale, zero_point

def asymmetric_dequantize(quantized_tensor, scale, zero_point):
	# Manual dequantization

	dequantized_tensor = scale * (quantized_tensor.float() - zero_point)
	return dequantized_tensor

def asymmetric_quantize(tensor, bit_width=8):
    # Define quantization range for int8: [-128, 127]
    q_min = -(2 ** (bit_width - 1))  # -128
    q_max = (2 ** (bit_width - 1)) - 1  # 127
    
    # Compute min and max of the input tensor
    x_min = tensor.min().item()
    x_max = tensor.max().item()
    
    # Calculate scale and zero-point
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = round(q_min - x_min / scale)
    
    # Ensure zero_point is within valid range
    zero_point = int(max(q_min, min(q_max, zero_point)))
    
    # Manual quantization: x -> q = round((x - z) / s)
    quantized_tensor = torch.round((tensor - zero_point) / scale).to(torch.int8)
    
    # Clamp to ensure values are within [q_min, q_max]
    quantized_tensor = torch.clamp(quantized_tensor, q_min, q_max)
    
    return quantized_tensor, scale, zero_point

def asymmetric_dequantize(quantized_tensor, scale, zero_point):
    # Manual dequantization: x_approx = s * (q - z)
    dequantized_tensor = scale * (quantized_tensor.float() - zero_point)
    return dequantized_tensor

def compute_mae(original, dequantized):
    # Compute Mean Absolute Error (MAE)
    error = original - dequantized
    mae = torch.mean(torch.abs(error)).item()
    return error.numpy(), mae

def plot_quantization(original, quantized, dequantized, error, mae):
    original_np = original.numpy()
    quantized_np = quantized.numpy()
    dequantized_np = dequantized.numpy()
    indices = np.arange(len(original_np))
    
    # Line Plot: Original, Quantized, Dequantized
    plt.figure(figsize=(12, 6))
    plt.plot(indices, original_np, 'o-', label='Original (float32)', color='blue', linewidth=2, markersize=8)
    plt.plot(indices, quantized_np, 's-', label='Quantized (int8)', color='red', linewidth=2, markersize=8)
    plt.plot(indices, dequantized_np, 'd-', label='Dequantized (float32)', color='green', linewidth=2, markersize=8)
    plt.title(f'Asymmetric Quantization and Dequantization\nMAE: {mae:.6f}', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Bar Plot: Side-by-side comparison
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    plt.bar(indices - bar_width, original_np, bar_width, label='Original (float32)', color='blue')
    plt.bar(indices, quantized_np, bar_width, label='Quantized (int8)', color='red')
    plt.bar(indices + bar_width, dequantized_np, bar_width, label='Dequantized (float32)', color='green')
    plt.title(f'Comparison of Values\nMAE: {mae:.6f}', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Error Plot
    plt.figure(figsize=(12, 6))
    plt.plot(indices, error, 'o-', label='Quantization Error', color='purple', linewidth=2, markersize=8)
    plt.axhline(y=mae, color='orange', linestyle='--', label=f'MAE: {mae:.6f}')
    plt.title('Quantization Error (Original - Dequantized)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Example tensor
tensor = torch.tensor([0.1, 0.5, 0.9, -0.2, 1.2], dtype=torch.float32)

# Quantize
quantized_tensor, scale, zero_point = asymmetric_quantize(tensor)
print("Original tensor:", tensor)
print("Quantized tensor:", quantized_tensor)
print("Scale:", scale)
print("Zero point:", zero_point)

# Dequantize
dequantized_tensor = asymmetric_dequantize(quantized_tensor, scale, zero_point)
print("Dequantized Tensor:", dequantized_tensor)

# Compute MAE and error
error, mae = compute_mae(tensor, dequantized_tensor)
print(f"\nMean Absolute Error (MAE): {mae:.6f}")

# Visualize
plot_quantization(tensor, quantized_tensor, dequantized_tensor, error, mae)

'''
Original tensor: tensor([ 0.1000,  0.5000,  0.9000, -0.2000,  1.2000])
Quantized tensor: tensor([-121,  -48,   25,   81,   80], dtype=torch.int8)
Scale: 0.0054901962771135215
Zero point: -92
Dequantized Tensor: tensor([-0.1592,  0.2416,  0.6424,  0.9498,  0.9443])
'''	