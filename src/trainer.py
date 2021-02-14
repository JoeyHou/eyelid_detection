import cv2 as cv
import cv2
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

import os

from utils import *


class Trainer():
    def __init__(self, trainer_config):

        self.data_identifier = trainer_config['data_identifier']
        self.curr_dir = trainer_config['curr_dir']

    def process_full_face_img(self):
        print('====================================')
        print(' => Processing full face images....')
        raw_data_dir = self.curr_dir + 'data/raw/full_img_' + self.data_identifier + '/'
        print('  => raw_data_dir:', raw_data_dir)

        eye_data_dir = self.curr_dir + 'data/processed/' + self.data_identifier + '/cropped/'
        left_data_dir = self.curr_dir + 'data/processed/' + self.data_identifier + '/left/'
        right_data_dir = self.curr_dir + 'data/processed/' + self.data_identifier + '/right/'
        os.system('mkdir '+ eye_data_dir)
        os.system('mkdir '+ left_data_dir)
        os.system('mkdir '+ right_data_dir)

        default_predictor_path = self.curr_dir + 'data/resources/shape_predictor_68_face_landmarks.dat'
        default_predictor = dlib.shape_predictor(default_predictor_path)

        img_info_lst = []
        error_img_lst = []
        counter = 0
        for fp in tqdm(listdir(raw_data_dir)):
            if '.png' not in fp:
                continue

            image = cv2.imread(raw_data_dir + fp)
            output, shape = predict_landmarks(image, default_predictor)
            shape = shape[36:48, :] # keep only eye part!
            counter += 1
            tmp_dic = {}
            tmp_dic['original_name'] = fp
            tmp_dic['img_idx'] = counter


            left_eye, right_eye = detect_eye(image, shape)
            try:
                cv2.imwrite(left_data_dir + 'left_' + str(counter) + '.png', left_eye)
                cv2.imwrite(right_data_dir + 'right_' + str(counter) + '.png', right_eye)
            except:
                print('failed on:', fp)

            # Cropping - keep two eyes
            inter_w = shape[6][0] - shape[3][0]
            left = int(shape[0][0] - inter_w / 2)
            right = int(shape[9][0] + inter_w / 2)
            up = min(shape[:, 1])
            down = max(shape[:, 1])
            inter_h = down - up
            up = up - inter_h
            down = down + inter_h
            output = output[up: down, left: right]
            image = image[up: down, left: right]
            shape = shape - np.array([left, up])
            tmp_dic['landmark'] = [tuple(i) for i in shape]

            # # Saving
            # if 0 in output.shape:
            #     error_img_lst.append(tmp_dic)
            #     continue
            # img_info_lst.append(tmp_dic)
            cv2.imwrite(eye_data_dir + str(counter) + '.png', output)
            #
            # # Cropping - seperate two eyes
            # tmp_dic['error_finding_eyes'] = 0
            # eyes = []
            # haarcascade_path = self.curr_dir + 'data/resources/haarcascade/'
            # eye1, eye2, out = detect_eye(image, fp, haarcascade_path)
            # if type(out) == int:
            #     tmp_dic['error_finding_eyes'] = 1
            # else:
            #     eyes.append(np.array([eye1, eye2]))
            #     cv2.imwrite(left_data_dir + 'left_' + str(counter) + '.png', eye2)
            #     cv2.imwrite(right_data_dir + 'right_' + str(counter) + '.png', eye1)
            img_info_lst.append(tmp_dic)
        return pd.DataFrame(img_info_lst)

    # def predict_eyelid_position(self):
    #
