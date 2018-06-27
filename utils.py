import os
import re
import sys
import json
import requests
import subprocess
from tqdm import tqdm
from contextlib import closing
from multiprocessing import Pool
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file
# Code based on https://github.com/keithito/tacotron/blob/master/util/audio.py
import math
import numpy as np
import numpy
import tensorflow as tf
from scipy import signal

import librosa
import librosa.filters


def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)

def normalize_Zscore(X_train, X_test):
    norm = '[ZM]'
    X=np.concatenate((X_train, X_test), axis=0)
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test

def normalize_MinMax(X_train, X_test):
    norm = '[MM]'
    X=np.concatenate((X_train, X_test), axis=0)
    X_min = np.min(X, axis = 0)
    X_max = np.max(X, axis = 0)
    X_train = (X_train - X_min) / (X_max-X_min)
    X_test = (X_test - X_min) / (X_max-X_min) 
    return X_train, X_test

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    return temp_batch
    
def dense_to_one_hot(labels_dense, num_classes=4):
    """Convert class labels from scalars to one-hot vectors"""
    c=np.max(labels_dense)+1
    labels_one_hot=np.eye(c)[labels_dense]
    return labels_one_hot

def batch_creator(batch_size, X_train, y_train,dim_input,n_classes):
    """Create batch with random samples and return appropriate format"""
    rng = np.random.RandomState(128)
    dataset_length=X_train.shape[0]
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = X_train[[batch_mask]].reshape(-1, dim_input)
    #batch_x = preproc(batch_x)
    
    batch_y = y_train[[batch_mask]]
    batch_y = dense_to_one_hot(batch_y,n_classes)
        
    return batch_x, batch_y