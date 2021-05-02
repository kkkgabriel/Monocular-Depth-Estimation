import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

'''
network mixin to inject the following functions:
- train_model: to train model
- validate_model: to get scores on validation set
- save_model: to save the model
- load_model: to load the model
'''
class NetMixin():

	"""
	Given a set of training data, this will train the weights of the model according to the dataset according to given hyperparameters.
	
	Parameters:
	- train_loader <Dataloader>: dataloader containing the the training dataset
	- val_loader <Dataloader>: dataloader containing the vlidation dataset
	- epoch <int>: number of iterations to run
	- optimiser <Optimizer>: optimizer that contains the model parameters to train, and the learning rate
	- loss_fn <torch.nn.modules.loss>: loss function that defines the loss between the prediction and the truth
	- device <str> (optional, default to 'cuda'): 'cuda' or 'cpu'; device to do training on
	- print_every <int> (optional, default to 1): number of eps to undergo before printing the logs
	- save_every <int> (optional, default to 1): number of eps to undergo before saving model to save_location
	- validate_every <int> (optional, default to 1): number of eps to undergo before validating model
	- save_location <str> (optional, default to 'models/new_model.pt'): location to save model

	Return: 
	- self: the model
	- loss: array of losses throughout the training
	"""
	def train_model(
		self,
		train_loader,
		val_loader,
		epochs,
		optimiser,
		loss_fn,
		device='cuda',
		print_every=1,
		save_every=1,
		validate_every=1,
		save_location='models/new_model.pt'
	):
		# init array of losses to store
		losses = []

		# set model to device and set training mode
		self.to(device)
		self.train()

		# training loop
		for epoch in range(epochs):

			training_loss = 0
			training_size = []

			# print epoch nunmber and date time if necessary
			if print_every > 0 and (epoch + 1) % print_every == 0:
				now = datetime.now()
				print(f'---------- Epoch {epoch + 1} of {epochs} @ {now.strftime("%d-%m-%Y %H:%M:%S")} ----------')

			# get loss value from training data, and step the weights
			for (X, y) in tqdm(train_loader):
				X, y = X.to(device), y.to(device)
				
				optimiser.zero_grad()
				output = self.forward(X)
				loss = loss_fn(output, y)

				training_loss += loss.item()
				training_size.append(X.size(0))
				loss.backward()
				optimiser.step()

			# print training loss if necessary
			if print_every > 0 and (epoch + 1) % print_every == 0:
				now = datetime.now()
				training_samples = sum(training_size)/training_size[0]
				print(f'Average training loss: {training_loss / training_samples}')

			# validate model if necessary
			if validate_every > 0 and (epoch + 1) % validate_every == 0:
				val_loss = self.validate_model(val_loader, loss_fn, device)

				print("Average Validation Loss: {:.3f}".format(val_loss))
				losses.append((epoch, training_loss / training_samples, val_loss))

			# save model if necessary
			if save_every > 0 and (epoch + 1) % save_every == 0:
				self.save_model(save_location)
			
			print()

		# save final model in location
		self.save_model(save_location)

		return losses


	def metrics_eval(self, pred, y, device='cuda', verbose=False):
		self.to(device)
		self.eval()

		pred = pred.to(device)
		y = y.to(device)
		pred = pred*255
		y = y*255
		pred = pred.cpu().detach().numpy()
		y = y.cpu().detach().numpy()
		thresh = np.maximum((y/pred),(pred/y))
		d1 = (thresh < 1.25).astype(int).mean()
		d2 = (thresh < 1.25**2).astype(int).mean()
		d3 = (thresh < 1.25**3).astype(int).mean()


		rms_log = (np.log(y) - np.log(pred)) ** 2
		rms_log = np.sqrt(rms_log.mean())

		abs_rel = np.mean(np.abs(y-pred)/y)
		sq_rel = np.mean((y-pred)**2/y)

		rms_log10 = np.sqrt(np.mean((np.log10(pred) - np.log10(y))**2))

		if verbose:
			print('------------y------------')
			print(y)
			print('\n---------pred-----------')
			print(pred)
			print('\n---------thresh-----------')
			print(thresh)
			print('\n---------thresh<1.25-----------')
			print(thresh<1.25)
			# print('\nResults: d1, d2, d3\n', d1, d2, d3, '\n')
			print('\nResults: d1, d2, d3, rms_log, abs_rel, sq_rel, rms_log10 \n', d1,d2,d3,rms_log,abs_rel,sq_rel,rms_log10, '\n')
		return d1, d2, d3, rms_log, abs_rel, sq_rel, rms_log10
		# return d1, d2, d3

	'''
	Get the loss values, and accuracy fromthe validation set

	Parameters:
	- val_loader <Dataloader>: dataloader containing the validation set
	- loss_fn <torch.nn.modules.loss>: loss function that defines the loss between the prediction and the truth
	- device <str>: device to run the validation on; 'cuda' or 'cpu' 

	Returns:
	- average loss: average loss for each data item
	- accuracy: percentage of number of correct predictions
	'''
	def validate_model(self, val_loader, loss_fn, device='cuda', advanced=False):
		# set model to evaluation mode
		self.to(device)
		self.eval()

		total_d1, total_d2, total_d3, total_rms_log, total_abs_rel, total_sq_rel, total_rms_log10  = 0, 0, 0, 0, 0, 0, 0

		training_loss = 0
		training_size = []

		with torch.no_grad(): # prevent weights from changing
			for (X, y) in tqdm(val_loader):
				X, y = X.to(device), y.to(device)
				output = self.forward(X)

				if advanced:
					d1, d2, d3, rms_log, abs_rel, sq_rel, rms_log10 = self.metrics_eval(output, y, device=device, verbose=False)
					total_d1 += d1
					total_d2 += d2
					total_d3 += d3
					total_rms_log += rms_log
					total_abs_rel += abs_rel
					total_sq_rel += sq_rel
					total_rms_log10 += rms_log10

				# compute average loss
				loss = loss_fn(output, y)
				training_loss += loss.item()

				training_size.append(X.size(0))
		self.train()

		training_samples = sum(training_size)/training_size[0]
		average_validation_loss = training_loss / training_samples
		
		# return advanced metrics if advanced set to true
		if advanced:
			avg_d1 = total_d1 / training_samples
			avg_d2 = total_d2 / training_samples
			avg_d3 = total_d3 / training_samples
			avg_rms_log = total_rms_log / training_samples
			avg_abs_rel = total_abs_rel / training_samples
			avg_sq_rel = total_sq_rel / training_samples
			avg_rms_log10 = total_rms_log10 / training_samples
			return average_validation_loss, avg_d1, avg_d2, avg_d3, avg_rms_log, avg_abs_rel, avg_sq_rel, avg_rms_log10

		return average_validation_loss

	'''
	test model using validate_model
	'''
	def test_model(self, test_loader, loss_fn, device='cuda', advanced=False):
		return self.validate_model(test_loader, loss_fn, device=device, advanced=advanced)
		 

	'''
	save the model to a path
	Parameters:
	- path <str>: location to save the model to
	'''
	def save_model(self, path):
		torch.save(self.state_dict(), path)

	
	'''
	load the model from a path
	Parameters:
	- path <str>: location to load the model from
	'''
	def load_model(self, path, device='cuda'):
		self.load_state_dict(torch.load(path, map_location=torch.device(device)))
		self.eval()