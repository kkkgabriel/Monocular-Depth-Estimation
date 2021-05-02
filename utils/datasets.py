from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import ndimage
from math import floor
from PIL import Image
import torch

'''
Dataset object for the nyu Depth Est V2 dataset
'''
class depth_est_dataset_v2(Dataset):
	def __init__(self, xy_filenames, data_zipfile):
		self.data_zipfile = data_zipfile
		self.xy_filenames = xy_filenames
	
		# set default transformation to be applied when dataloader calls this
		self.tx = transforms.Compose([transforms.ToTensor()])

	def set_tx(self, tx):
		"""
		Setter function for transformations

		Parameters:
		- 
		"""
		self.tx = tx

	def open_images(self, idx):
		"""
		Opens image with specified parameters.
		"""
		# get filenames
		x_filename, y_filename = self.xy_filenames[idx]

		# get input images
		x = self.data_zipfile.open(x_filename)
		x = Image.open(x)
		x = np.array(x, dtype='uint8' )

		# get target depthmap
		y = self.data_zipfile.open(y_filename)
		y = Image.open(y)
		y = np.array(y, dtype='uint8' )

		return x, y

	def show_images(self, idx):
		"""
		Opens, then displays image with specified parameters.
		"""

		# Open image
		image, depth_map = self.open_images(idx)

		# Display
		f, axarr = plt.subplots(1,2, figsize=(24,32))
		axarr[0].imshow(image, cmap='gray')
		axarr[1].imshow(depth_map, cmap='gray')
		axarr[0].axis('off')
		axarr[1].axis('off')

	def __len__(self):
		"""
		magic function that will be called by dataloader
		"""
		return len(self.xy_filenames)

	def __getitem__(self, idx):
		"""
		magic function that will be called by dataloader to return x and y for model usage 
		"""
		# get cls of object using data_indexes created in children classes
		x, y = self.open_images(idx)

		# execute transformations
		x = self.tx(x)
		y = transforms.ToTensor()(y) # convert y to tensor

		return x, y

'''
Dataset object for the nyu Depth Est V1 dataset
'''
class depth_est_dataset(Dataset):
	def __init__(self, filename, start, end):
		# convert file into hf
		hf = h5py.File(filename, 'r')

		# convert fraction to indexes
		start = floor(len(hf['images']) * start)
		end = floor(len(hf['images']) * end)

		self.inputs = hf['images'][start:end]
		self.targets = hf['depths'][start:end]
		self.hf = hf

		# set default transformation to be applied when dataloader calls this
		self.tx = transforms.Compose([transforms.ToTensor()])

	def set_tx(self, tx):
		"""
		Setter function for transformations

		Parameters:
		- 
		"""
		self.tx = tx
		
	def open_images(self, idx):
		"""
		Opens image with specified parameters.
		"""
		# get raw images
		image = self.inputs[idx].reshape((640, 480, 3))
		image = ndimage.rotate(image, -90)

		# get target depthmap
		depth_map = self.targets[idx]
		depth_map = ndimage.rotate(depth_map, -90)

		return image, depth_map

	def show_images(self, idx):
		"""
		Opens, then displays image with specified parameters.
		"""
		
		# Open image
		image, depth_map = self.open_images(idx)
		
		# Display
		f, axarr = plt.subplots(1,2)
		axarr[0].imshow(image)
		axarr[1].imshow(depth_map)

	def __len__(self):
		"""
		magic function that will be called by dataloader
		"""
		assert len(self.targets) == len(self.inputs)
		return len(self.targets)

	def __getitem__(self, idx):
		"""
		magic function that will be called by dataloader to return x and y for model usage 
		"""
		# get cls of object using data_indexes created in children classes
		x, y = self.open_images(idx)

		# execute transformations
		x = self.tx(x)
		y = transforms.ToTensor()(y) # convert y to tensor

		return x, y

'''
'''
def get_datasets(data_filename, ratio):
	train_end = val_start = ratio[0]/sum(ratio)
	val_end = test_start = sum(ratio[0:2])/sum(ratio)

	train = depth_est_dataset(data_filename, 0, train_end)
	val = depth_est_dataset(data_filename, val_start, val_end)
	test = depth_est_dataset(data_filename, test_start, 1)

	return {'train': train, 'val': val, 'test': test}

'''
utility function to load a full dataset into train, validation, and test dataloader

Parameters
- datasets <dict>: to contain 'train', 'test', and 'validation' keys which have the correspoding datasets
- params <dict>: to contain the parameters for the dataloader
'''
def get_dataloaders(datasets, params):
	trainloader = DataLoader(datasets['train'], **params)
	testloader = DataLoader(datasets['test'], **params)
	validloader = DataLoader(datasets['val'], **params)
	return trainloader, validloader, testloader

'''
Display predictions and ground truth
'''
def display_predictions(model, x, y, loss_fn, cmap='gray'):
	model.cpu()
	model.eval()

	x = torch.unsqueeze(x, 0)
	prediction = model.forward(x)
	prediction = torch.squeeze(prediction, 0)

	# compute loss
	loss = loss_fn(prediction, y)
	print('Loss value: {}'.format(loss.item()))

	x = x.cpu()
	x = torch.squeeze(x, 0)
	x = x.permute(1,2,0)
	x = np.array(x)
	prediction = np.array(prediction.detach().cpu())
	prediction = prediction.reshape((480, 640))
	y = np.array(y.cpu())
	y = y.reshape((480, 640))

	f, axarr = plt.subplots(1, 3, figsize=(24,32))
	# set cmap
	axarr[0].imshow(x, cmap=cmap)
	axarr[1].imshow(prediction, cmap=cmap)
	axarr[2].imshow(y, cmap=cmap)

	# make axes go away
	axarr[0].axis('off')
	axarr[1].axis('off')
	axarr[2].axis('off')

	# change titles
	axarr[0].title.set_text("Input image")
	axarr[1].title.set_text("Predicted depthmap")
	axarr[2].title.set_text("True depthmap")
	plt.show()