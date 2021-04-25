import torch.nn as nn	
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from resnet_model import block, ResNet, ResNet50




device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
writer = SummaryWriter('run/trial')

writer.add_scalar(tag='hi', scalar_value= 0, global_step= 0)

model = ResNet50().to(device)

x = torch.randn(size= (2, 3,224,224)).to(device)

out = model(x)

print(out)