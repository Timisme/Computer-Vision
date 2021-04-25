import os, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from utils.dataset import Dataset
from utils.model import ResNet, block
import cv2


class Show_Image():
		def __init__(self, data_dir, model_dir):

			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.model = torch.load(model_dir).to(self.device)
			self.model.eval()
			self.test_dataset, self.inputs = self.split_Train_Val_Data(data_dir= data_dir)
			self.cls_dict = {0:'cat', 1:'dog'}

		def split_Train_Val_Data(self, data_dir):

			normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

			test_transformer = transforms.Compose([
				transforms.Resize(224),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize
			])
			
			dataset = ImageFolder(data_dir) 
			
			character = [[] for i in range(len(dataset.classes))]
			
			# 將每一類的檔名依序存入相對應的 list
			for x, y in dataset.samples:
				character[y].append(x)
			  
			test_inputs, test_labels = [], []

			for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
				
				np.random.seed(42)
				np.random.shuffle(data)
				
				num_sample_train = 0.8
				num_sample_test = 0.2
					
				for x in data[int(len(data)*num_sample_train):] : # 後 20% 資料存進 testing list
					try : 
						img = Image.open(x).convert('RGB')
						test_inputs.append(x)
						test_labels.append(i)
					except: 
						continue
			test_dataset = Dataset(test_inputs, test_labels, test_transformer)

			return test_dataset, test_inputs

		def show_img(self):

			idx = torch.randint(low= 1, high= len(self.inputs), size= (1, )).item()

			if idx > len(self.inputs):
				raise ValueError('please enter a integer smaller than {}'.format(len(self.inputs)))
			else: 
				with torch.no_grad():

					img, label = self.test_dataset[idx]

					img = torch.unsqueeze(img, dim=0)

					outputs = self.model(img.to(self.device))

				_, predicted = torch.max(outputs.data,1)

				img = cv2.imread(self.inputs[idx], 3)

				b,g,r = cv2.split(img)           # get b, g, r
				rgb_img = cv2.merge([r,g,b]) 
				
				cv2.imshow(f'{self.cls_dict[predicted.item()]}',rgb_img)
				# plt.title(f'{self.cls_dict[predicted.item()]}')
				# plt.axis('off')
				plt.show()
		def show_tensorboard(self):

			img = plt.imread(fname='data/training_accu.png')
			plt.imshow(img)
			plt.show()
		def show_comparision(self):
			img = plt.imread(fname='data/vision_comparision.png')
			plt.imshow(img)
			plt.show()

if __name__ == '__main__':
	
	data_dir = '../data/dataset/PetImages/'
	model_dir = '../trained_model/ResNet_hw2.pt'

	show_img = Show_Image(data_dir= data_dir, model_dir= model_dir)
	show_img.show_img()


	

