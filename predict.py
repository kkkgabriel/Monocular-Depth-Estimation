# MUST RUN THIS FIRST!!!

#---------- IMPORTS -----------#

# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models

# data analytics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# for image and dataset loading
from PIL import Image
from imageio import imread
from scipy import ndimage

# misc
import re
from datetime import datetime
import argparse
from zipfile import ZipFile

# custom code
from utils.datasets import *
from utils.config import *
from utils.network import *
from utils.netMixin import *
from utils.attacks import *

import warnings
warnings.filterwarnings("ignore")

#---------- get args ----------#
parser = argparse.ArgumentParser(description="Predict depth on a sample image from the test set.")
parser.add_argument("model_location", help="Location model")
parser.add_argument("index", type=int, help="index of sample in testset to run test on")
parser.add_argument("--device", default='cpu', help='cuda or cpu')

args = parser.parse_args()
model_location = args.model_location
index  = args.index
device = args.device

# init the torch seeds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)

#---------- preparing the data ----------#
# file dirs
data_filename = 'data/nyu_data.zip'
train_csv_filename = 'data/nyu2_train.csv'
test_csv_filename = 'data/nyu2_test.csv'

# extract filenames from zip file
zf = ZipFile(data_filename)
train_data = zf.read(train_csv_filename).decode("utf-8") 
train_data = np.array([line.split(',') for line in train_data.split()])

# define train validation set split ratio
train_val_test_ratio = [0.7, 0.1, 0.2]

# get (train, val, test) start and end indices
_, train_end, val_end = [sum(train_val_test_ratio[:n]) for n in range(len(train_val_test_ratio))]
train_end = val_start = floor(train_end * len(train_data))
val_end = test_start = floor(val_end * len(train_data))

# train_data contains paths to xs and ys, spliting it for train, val and test
train_xy = train_data[:train_end]
val_xy = train_data[val_start:val_end]
test_xy = train_data[test_start:]

if index < 0:
	print('ERROR: Input a index > 0')
	exit()

if index > len(test_xy):
	print('ERROR: Input a index, where 0 > index > {}'.format(len(test_xy)))
	exit()


#--------- get trained model ----------#
trained_model = depth8sig()
trained_model.load_model(model_location, device=device)
criterion = depthEstLossLog()

#----------- init testset------------#
testset = depth_est_dataset_v2(test_xy, zf)
x, y = testset[index]
display_predictions(trained_model, x, y, criterion)