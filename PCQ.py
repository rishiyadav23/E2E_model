import torch

def quantization_error(tensor, dequantized_tensor):
    return (dequantized_tensor - tensor).abs().square().mean()

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

def symmetric_dequantize(quantized_tensor, scale):
	# Manual dequantization

	dequantized_tensor = scale * quantized_tensor.float()
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

test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

# Along the rows 
quantized_tensor0, scale0 = linear_q_symmetric_per_channel(test_tensor, dim=0)

# Along the columns
quantized_tensor1, scale1 = linear_q_symmetric_per_channel(test_tensor, dim=1)

dequantized_tensor0 = symmetric_dequantize(quantized_tensor0, scale0)

dequantized_tensor1 = symmetric_dequantize(quantized_tensor1, scale1)

print(f"""Quantization Error : \
{quantization_error(test_tensor, dequantized_tensor0)}""")

print(f"""Quantization Error : \
{quantization_error(test_tensor, dequantized_tensor1)}""")