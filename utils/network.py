import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.netMixin import NetMixin

'''
8 layer encoder-decoder
'''
class depth8(nn.Module, NetMixin):
	def __init__(self, hidden_layer=8):
		super(depth8, self).__init__()
		self.hidden_layer = hidden_layer
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 128, 4, stride=2, padding=2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(128, 128, 4, stride=2, padding=2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(128, 64, 4, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 64, 4, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 32, 2, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(32, 32, 2, stride=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(32, 16, 2, stride=1),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(16, hidden_layer, 2, stride=1),
			nn.BatchNorm2d(hidden_layer),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.decoder = nn.Sequential(
			# 1
			nn.ConvTranspose2d(hidden_layer, 8, 4, stride = 2, padding = 0, output_padding = 0, dilation=1),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(0.2, inplace=True),
			# 2
			nn.ConvTranspose2d(8, 12, 4, stride = 1, padding = 2, output_padding = 0, dilation=1),
			nn.BatchNorm2d(12),
			nn.LeakyReLU(0.2, inplace=True),
			# 3
			nn.ConvTranspose2d(12, 16, 4, stride = 2, padding = 1, output_padding = 0, dilation=2),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.2, inplace=True),
			# 4
			nn.ConvTranspose2d(16, 16, 4, stride = 1, padding = 2, output_padding = 0, dilation=1),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.2, inplace=True),
			# 5
			nn.ConvTranspose2d(16, 32, 4, stride = 2, padding = 1, output_padding = 1, dilation=2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),
			# 6
			nn.ConvTranspose2d(32, 64, 4, stride = 1, padding = 1, output_padding = 0, dilation=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			# 7
			nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 0, output_padding = 1, dilation=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			# 8
			nn.ConvTranspose2d(64, 1, 4, stride = 1, padding = 0, output_padding = 0, dilation=1),
			nn.BatchNorm2d(1),
			nn.LeakyReLU(0.2, inplace=True),
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

'''
RMSE Loss 
'''
class RMSELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()
		
	def forward(self,yhat,y):
		return torch.sqrt(self.mse(yhat,y))

'''
Depth Est loss
'''
class depthEstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self,yhat,y):
        yhat = yhat*255
        y = y*255
        no_ele = torch.numel(y)
        return self.mse(yhat,y)-((torch.sum(yhat-y)**2)*0.5/(no_ele**2))
'''
Depth Est loss log
'''
class depthEstLossLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self,yhat,y):
        # print('this shit')
        # print(torch.min(y))
        yhat = yhat*255
        y = y*255
        lam = 1e-1
        # print(yhat)
        # print(y)
        # print(yhat.shape)
        # print(y.shape)
        # yhat= yhat + lam
        # y = y + lam
        # checkyhat = yhat<0
        # checky = y<0
        # print(torch.all(checky))
        # print(torch.all(checkyhat))
        # print(torch.min(yhat))
        # print('changed')
        # print(lam)
        # print(torch.max(torch.log(yhat)))
        # print(torch.max(torch.log(y)))
        no_ele = torch.numel(y)
        # return self.mse(torch.log(yhat),torch.log(y))-((torch.sum(torch.log(yhat)-torch.log(y))**2)*0.5/(no_ele**2))
        
        
        d = torch.log(yhat)-torch.log(y)
        # print(no_ele)
        # print('this is d')
        # print(d)
        # help = torch.clamp(d,min = 0,max = 2.406)
        # print(help)
        # print(torch.all(help<2.5))
        # print(torch.max(help))
        # print(torch.sum(help))
        # print('this is nan')
        # print((torch.sum(d)**2))
        # print('sdfafaf')
        # print((torch.sum(d)**2)*0.5/(no_ele**2))
        return self.mse(torch.log(yhat),torch.log(y))-((torch.sum(d)**2)*0.5/(no_ele**2))
        