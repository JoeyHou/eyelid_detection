# import cv2 as cv
from os import listdir
import math
import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import pandas as pd
from scipy import optimize
from tqdm import tqdm
import seaborn as sns
import dlib
import random
import xml.etree.ElementTree as ET


file_dir = './'

lid_data_dir = file_dir + '../data/lid_data/'
lid_model_dir = file_dir + '../models/lid_models/'


curr_success = []
curr_eye = 'left'
all_xml_file_path = file_dir + '../data/lid_data/' + curr_eye + '/' + curr_eye + '_eyes_11points_all.xml'

empty_xml_file_path = file_dir + '../data/lid_data/' + curr_eye + '/' + curr_eye + '_eyes_11points_empty.xml'
train_xml_file_path = file_dir + '../data/lid_data/' + curr_eye + '/' + curr_eye + '_eyes_11points_train.xml'
test_xml_file_path = file_dir + '../data/lid_data/' + curr_eye + '/' + curr_eye + '_eyes_11points_test.xml'
val_xml_file_path = file_dir + '../data/lid_data/' + curr_eye + '/' + curr_eye + '_eyes_11points_val.xml'

predictor_path = lid_model_dir + curr_eye + '_11_eyes_predictor.dat'

# import the necessary packages
import multiprocessing

def train(parameters):
#     print("===========================================")
#     print("[INFO] setting shape predictor options...")
    options = dlib.shape_predictor_training_options()
#     print(parameters)
    # [2-8]
    options.tree_depth = parameters['tree_depth']

    # regularization parameter 
    # [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8]
    options.nu = parameters['nu']

    # [6, 18]
    options.cascade_depth = parameters['cascade_depth']

    # Larger for more accurate results & Slower inference
    options.feature_pool_size = parameters['feature_pool_size']

    # Larger for more accurate results & Slower training
    options.num_test_splits = parameters['num_test_splits']

    # controls amount of "jitter" (i.e., data augmentation) when training
    # the shape predictor -- applies the supplied number of random
    # deformations, thereby performing regularization and increasing the
    # ability of our model to generalize
    # [0, 50] 
    options.oversampling_amount = parameters['oversampling_amount']

    # amount of translation jitter to apply -- the dlib docs recommend
    # values in the range [0, 0.5]
    options.oversampling_translation_jitter = parameters['oversampling_translation_jitter']

    # tell the dlib shape predictor to be verbose and print out status
    # messages our model trains
    options.be_verbose = True

    # number of threads/CPU cores to be used when training -- we default
    # this value to the number of available cores on the system, but you
    # can supply an integer value here if you would like
    # options.num_threads = multiprocessing.cpu_count()
    options.num_threads = 2
    
    
    # train the shape predictor
    # print("[INFO] training shape predictor...")
    dlib.train_shape_predictor(train_xml_file_path, predictor_path, options)



first_trial = pd.read_csv(lid_model_dir + 'archived/Copy of first_trial.csv')
second_trial = pd.read_csv(lid_model_dir + 'archived/Copy of second_trial.csv')
third_trial = pd.read_csv(lid_model_dir + 'archived/Copy of third_trial.csv')
all_trials = pd.concat([first_trial, second_trial, third_trial]).sort_values('error').reset_index(drop = True)

performance_lst = []
for i in tqdm(range(30)):
    curr_model = dict(all_trials.iloc[i])
    curr_model['tree_depth'] = int(curr_model['tree_depth'])
    curr_model['cascade_depth'] = int(curr_model['cascade_depth'])
    curr_model['feature_pool_size'] = int(curr_model['feature_pool_size'])
    curr_model['num_test_splits'] = int(curr_model['num_test_splits'])
    curr_model['oversampling_amount'] = int(curr_model['oversampling_amount'])
    train(curr_model)
    error = dlib.test_shape_predictor(val_xml_file_path, predictor_path)
    curr_model['error'] = error
    performance_lst.append(curr_model)