import numpy as np
import torch

class transform_enm_attack(object):
	def __init__(self, eps= 0.25):
		self.eps = eps
		
	def __call__(self, tensor):
		img_rows = tensor.shape[-2]
		img_cols = tensor.shape[-1]
		epsilon_mat = np.asarray(([[2*(np.random.normal() - 0.5) * self.eps 
									for i in range(img_cols)]
									for j in range(img_rows)]))      
	  
		eps_image = tensor + epsilon_mat
		eps_image = eps_image.float()
		eps_image = torch.clamp(eps_image, 0, 1)

		return eps_image    

	def __repr__(self):
		return self.__class__.__name__ + '(eps ={0.25})'.format(self.eps)


class transform_fgsm_attack(object):
	def __init__(self, model, testloader, criterion, eps=0.25):
		self.eps = eps

		counter = 0
		pertubed_image_set = []
		data_set = []

		for data, target in testloader:
			counter +=1
			#data, target = data.to(device), target.to(device)
			data.requires_grad = True
			output = model(data)
			
			# Calculate the loss
			loss = criterion(output,target)

			# Zero all existing gradients
			model.zero_grad()

			# Calculate gradients of model in backward pass
			loss.backward()

			# Collect datagrad
			data_grad = data.grad.data
			pertubed_image_set.append(data_grad)
			data_set.append(data)
			
			# limit to 16 as getting gradient of all test set takes too long + runtime crashes when loading more than 20 
			break
		self.data_set = data_set
		self.pertubed_image_set = pertubed_image_set

	def getgrad(self, x):
		# get data_grad from list here
		a = self.data_set[0].tolist()
		b = a.index(x.tolist())
		c = self.pertubed_image_set[0][b]  
		return c
		
	def __call__(self, data):
		#tensor= tensor.to(device), target.to(device)
		data_grad = self.getgrad(data)

		sign_data_grad = data_grad[0].sign()
		perturbed_image = data + self.eps*sign_data_grad
		perturbed_image = torch.clamp(perturbed_image, 0, 1)
		# print(sign_data_grad)
		return perturbed_image    
	  
	def __repr__(self):
		return self.__class__.__name__ + '(eps ={0.25})'.format(self.eps)

class targetted_object_noise_sample24(object):
	def __init__(self, eps=0.25, verbose=False):
		self.eps = eps
		self.verbose = verbose
		
	def __call__(self, tensor):
		img_rows = tensor.shape[-2]
		img_cols = tensor.shape[-1]
		epsilon_mat = np.asarray(([[ np.random.normal()*(self.eps/8)
									for i in range(img_cols)]
									for j in range(img_rows)]))  
		epsilon_mat[150:400, 350:550] = [[np.random.normal()*self.eps for i in range (200)] for i in range(250)]

		if self.verbose:
			print('epsilon has values?: ', not np.all(epsilon_mat<=0))
			if np.all(epsilon_mat<=0):
				print('OH NO ')
				print(epsilon_mat)

		eps_image = tensor + epsilon_mat
		eps_image = eps_image.float()
		eps_image = torch.clamp(eps_image, 0, 1)

		return eps_image    

	def __repr__(self):
		return self.__class__.__name__ + '(eps ={0.25})'.format(self.eps)


class targetted_object_noise_sample30(object):
	def __init__(self, eps=0.25, verbose=False):
		self.eps = eps
		self.verbose = verbose
		
	def __call__(self, tensor):
		img_rows = tensor.shape[-2]
		img_cols = tensor.shape[-1]
		epsilon_mat = np.asarray(([[ np.random.normal()*(self.eps/8)
								  for i in range(img_cols)]
								  for j in range(img_rows)]))  
		epsilon_mat[0:200, 250:425] = [[np.random.normal()*self.eps for i in range (175)] for i in range(200)]

		if self.verbose:
			print('epsilon has values?: ', not np.all(epsilon_mat<=0))
			if np.all(epsilon_mat<=0):
				print('OH NO ')
				print(epsilon_mat)

		eps_image = tensor + epsilon_mat
		eps_image = eps_image.float()
		eps_image = torch.clamp(eps_image, 0, 1)

		# Return
		return eps_image 

	def __repr__(self):
		return self.__class__.__name__ + '(eps ={0.25})'.format(self.eps)