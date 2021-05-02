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
parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("model_location", help="Location of first layer model")
parser.add_argument("--device", default='cpu', help="Device to test on, default at 'cpu'")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to run test on, default at '16'")

args = parser.parse_args()
model_location = args.model_location
batch_size = args.batch_size
device = args.device

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

#--------- get trained model ----------#
trained_model = depth8()
trained_model.load_model(model_location, device=device)
criterion = depthEstLoss()


#--------- init dataloaders for testing ----------#
datasets = {
    'train': depth_est_dataset_v2(train_xy, zf),
    'val': depth_est_dataset_v2(val_xy, zf),
    'test': depth_est_dataset_v2(test_xy, zf)
}
_, _, testloader = get_dataloaders(datasets, {'batch_size': batch_size})

# Test the model and get results
advanced = True # get advanced metrics instead of just test loss
scores = trained_model.test_model(testloader, criterion, advanced=advanced, device=device)
metrics = ['Average Test loss', 'avg_d1', 'avg_d2', 'avg_d3', 'avg_rms_log', 'avg_abs_rel', 'avg_sq_rel', 'avg_rms_log10']
print(pd.DataFrame(data={'Metrics': metrics, 'Scores':scores}))