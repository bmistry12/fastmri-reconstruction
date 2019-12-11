import numpy as np
import torch
import h5py, os
from functions import transforms as T
from functions.subsample import MaskFunc
from scipy.io import loadmat
from torch.utils.data import DataLoader
from skimage.measure import compar_ssim
from matplotlib import pyplot as plt

train_data_path = "C:\Users\j-ayu\OneDrive\Documents\University\Uni Year 3\Neural Computation\Coursework\Datasets\Train"

# For mask 4AF - acc = 4, cen = 0.08
# For mask 8AF - acc = 8, cen = 0.04
acc = 4
cen_fract = 0.08
seed = True
num_workers = 12

def load_data_path(train_data_path):
