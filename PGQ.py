import torch

def scale_quantize(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max
    scale = r_max / q_max
    return scale

def quantized(tensor, scale, dtype=torch.int8):
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

	# Quantization
    quantized_tensor = torch.round(tensor / scale).to(dtype)

	# Clamp to ensure values are within range
    quantized_tensor = quantized_tensor.clamp(q_min, q_max).to(dtype)

    return quantized_tensor

def quantization_error(tensor, dequantized_tensor):
    return (dequantized_tensor - tensor).abs().square().mean()

def linear_quantized_per_group(tensor, group_size, dtype=torch.int8):
    
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2

    tensor = tensor.view(-1, group_size)
    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)
    quantized_tensor = quantized_tensor.view(t_shape)

    return quantized_tensor, scale

def symmetric_dequantize(quantized_tensor, scale):
	# Manual dequantization

	dequantized_tensor = scale * quantized_tensor.float()
	return dequantized_tensor

def linear_dequantized_per_group(quantized_tensor, scale, group_size):
	q_shape = quantized_tensor.shape
	quantized_tensor = quantized_tensor.view(-1, group_size)
	dequantized_tensor = symmetric_dequantize(quantized_tensor, scale)
	dequantized_tensor = dequantized_tensor.view(q_shape)

	return dequantized_tensor

def linear_q_symmetric_per_channel(tensor, dim, dtype=torch.int8):
	output_dim = tensor.shape[dim]
	scale = torch.zeros(output_dim)

	for index in range(output_dim):
		subtensor = tensor.select(dim, index)
		scale[index] = scale_quantize(subtensor, dtype)

	# Reshape scale
	scale_shape = [1]*tensor.dim()
	scale_shape[dim] = -1
	scale = scale.view(scale_shape)
	quantized_tensor = quantized(tensor, scale, dtype)

	return quantized_tensor, scale

test_tensor = torch.rand((6, 6))

group_size = 3

# Along the rows 
quantized_tensor, scale = linear_quantized_per_group(test_tensor, group_size=group_size)

# Along the columns
dequantized_tensor = linear_dequantized_per_group(quantized_tensor, scale, group_size=group_size)

print(f"""Quantization Error : \
{quantization_error(test_tensor, dequantized_tensor)}""")