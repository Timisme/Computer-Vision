import torch 
import torch.nn as nn

class block(nn.Module):
	def __init__(self, in_channels, out_channels, downsample= None, stride= 1):
		super(block, self).__init__()
		self.expansion = 4
		self.conv1=  nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 1, stride= 1, padding= 0)
		self.bn1= nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, stride= stride, padding= 1)
		self.bn2= nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels*self.expansion, kernel_size= 1, stride= 1, padding= 0)
		self.bn3= nn.BatchNorm2d(out_channels*self.expansion)
		self.relu= nn.ReLU()
		self.downsample = downsample

	def forward(self, x):

		identity= x

		x= self.conv1(x)
		x= self.bn1(x)
		x= self.relu(x)
		x= self.conv2(x)
		x= self.bn2(x)
		x= self.relu(x)
		x= self.conv3(x)
		x= self.bn3(x)

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = self.relu(x)
		return x 


class ResNet(nn.Module): # [3, 4, 6, 3]
	def __init__(self, block, layers, img_channels, n_class):
		super(ResNet, self).__init__()
		self.expansion= 4
		self.in_channels= 64
		self.conv1= nn.Conv2d(in_channels= img_channels, out_channels= self.in_channels, kernel_size= 7, stride= 2, padding= 3)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
		
		self.layer1 = self.ResNet_layer(block= block, n_res_blocks= layers[0], out_channels= 64, stride=1)
		self.layer2 = self.ResNet_layer(block= block, n_res_blocks= layers[1], out_channels= 128, stride=2)
		self.layer3 = self.ResNet_layer(block= block, n_res_blocks= layers[2], out_channels= 256, stride=2)
		self.layer4 = self.ResNet_layer(block= block, n_res_blocks= layers[3], out_channels= 512, stride=2)

		self.avgpool= nn.AdaptiveAvgPool2d((1,1))
		self.fc = nn.Linear(512*4, n_class)

	def forward(self, x):

		x= self.conv1(x)
		x= self.bn1(x)
		x= self.relu(x)
		x= self.maxpool(x)

		x= self.layer1(x)
		x= self.layer2(x)
		x= self.layer3(x)
		x= self.layer4(x)

		x= self.avgpool(x)
		x = x.reshape(x.shape[0], -1)
		x= self.fc(x)

		return x



	def ResNet_layer(self, block, n_res_blocks, out_channels, stride):
		downsample = None
		layers = []

		if stride != 1 or self.in_channels != out_channels * self.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size= 1, stride= stride),
			nn.BatchNorm2d(out_channels*self.expansion))

		layers.append(block(self.in_channels, out_channels, downsample, stride))
		self.in_channels= out_channels*self.expansion

		for i in range(n_res_blocks- 1):
			layers.append(block(self.in_channels, out_channels))

		return nn.Sequential(*layers)

def ResNet50(img_channels= 3, n_class= 10):
	return ResNet(block, [3, 4, 6, 3], img_channels= img_channels, n_class= n_class)