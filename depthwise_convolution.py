import torch
import torch.nn as nn

#Regular Convolution
conv = nn.Conv2d(in_channels = 10,out_channels = 32,kernel_size = 3)

#Depthwise Separable Convolution
depth_conv = nn.Conv2d(in_channels = 10,out_channels = 10,kernel_size = 3, groups = 10)
point_conv = nn.Conv2d(in_channels = 10,out_channels = 32,kernel_size = 1)
depthwise_separable_convolution = nn.Sequential(depth_conv,point_conv)

if __name__ == '__main__':
	input_tensor = torch.rand(5,10,50,50)
	out_regular_conv = conv(input_tensor)
	params_regular_convolution = sum(p.numel() for p in conv.parameters() if p.requires_grad)
	out_depthwise_convolution = depthwise_separable_convolution(input_tensor)
	params_depthwise_convolution = sum(p.numel() for p in depthwise_separable_convolution.parameters() if p.requires_grad)
	print(f'Regular convolution needs {params_regular_convolution} parameters')
	print(f'Depthwise Separable Convolution needs {params_depthwise_convolution} parameters')
	size_regular_convolution = out_regular_conv.size()
	size_depthwise_separable_convolution = out_depthwise_convolution.size()
	print(size_regular_convolution)
	assert (size_regular_convolution == size_depthwise_separable_convolution)

