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

# init the torch seeds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)

#---------- get args ----------#
parser = argparse.ArgumentParser(description="Train depth estimation model")
parser.add_argument("--save_location", default='models/new_model.pt', help="Location to save model")
parser.add_argument("--hidden_layers", default=8, type=int, help="Number of hidden_layers (channels)")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight Decay")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to train")
parser.add_argument("--device", default='cpu', help='Device to train on')

args = parser.parse_args()
save_location = args.save_location
hidden_layers = args.hidden_layers
lr = args.learning_rate
eps = args.epochs
device = args.device
weight_decay = args.weight_decay

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

#-------- init model for training --------#
model = depth8sig(hidden_layers)

#--------- init dataloaders for training ----------#
datasets = {
    'train': depth_est_dataset_v2(train_xy, zf),
    'val': depth_est_dataset_v2(val_xy, zf),
    'test': depth_est_dataset_v2(test_xy, zf)
}
trainloader, validloader, _ = get_dataloaders(datasets, {'batch_size': 16})

#--------- set hyperparams --------------#
criterion = depthEstLossLog()
optimizer = torch.optim.RMSprop(model.parameters(), weight_decay=weight_decay, lr=lr)

#-------- training loop ----------#
# Note:
# This training loop is just a sample to train your own model.
# Our model, models/Depth8.pt is trained with this function with 15 eps.

losses = model.train_model(
    trainloader,
    validloader,
    eps,
    optimizer,
    criterion,
    save_location=save_location,
    device=device
)