import cv2
import dlib

import math
import numpy as np
import pandas as pd
from scipy import optimize
import seaborn as sns
# from matplotlib.pyplot import imshow
# from matplotlib import pyplot as plt

import os
from tqdm import tqdm

from utils import *


class Trainer():
    def __init__(self, config):

        self.data_sources = config['data_sources']
        self.compare_with_manual_measurement = config['compare_with_manual_measurement']
        self.model_name = config['model_name']

        if config['curr_dir'] == "":
            self.curr_dir = './'
        else:
            self.curr_dir = config['curr_dir']

        if config['processed_eye_dir'] == "":
            self.processed_eye_dir = self.curr_dir + 'data/processed/' + self.model_name
        else:
            self.processed_eye_dir = self.curr_dir + 'data/processed/' + config['processed_eye_dir']
        os.system('mkdir -p ' + self.processed_eye_dir)

        if config['final_output_dir'] == "":
            self.final_output_dir = self.curr_dir + 'data/final_output/' + self.model_name
        else:
            self.final_output_dir = self.curr_dir + 'data/final_output/' + config['final_output_dir']
        os.system('mkdir -p ' + self.final_output_dir)

        if config['mrd_predictor_dir'] == "":
            self.mrd_predictor_dir = self.curr_dir + '/models'
        else:
            self.mrd_predictor_dir = self.curr_dir + '/models/' + config['mrd_predictor_dir']

    def process_full_face_img(self):
        print('====================================')
        print(' => Processing full face images....')
        all_meta_data = []
        for data_source in self.data_sources:
            tmp_df = self.handle_single_data_source(data_source)
            tmp_df['data_source'] = data_source
            all_meta_data.append(tmp_df)
        return pd.concat(all_meta_data).reset_index(drop = True)

    def handle_single_data_source(self, data_source):
        raw_data_dir = self.curr_dir + 'data/raw/' + data_source + '/'
        print('  => raw_data_dir:', raw_data_dir)

        eye_data_dir = self.processed_eye_dir + '/cropped/'
        left_data_dir = self.processed_eye_dir + '/left/'
        right_data_dir = self.processed_eye_dir + '/right/'
        os.system('mkdir -p '+ eye_data_dir)
        os.system('mkdir -p '+ left_data_dir)
        os.system('mkdir -p '+ right_data_dir)

        default_predictor_path = self.curr_dir + 'data/resources/shape_predictor_68_face_landmarks.dat'
        default_predictor = dlib.shape_predictor(default_predictor_path)

        img_info_lst = []
        error_img_lst = []
        counter = 0
        for fp in tqdm(os.listdir(raw_data_dir)):
            if '.png' not in fp:
                continue

            image = cv2.imread(raw_data_dir + fp)
            output, shape = predict_landmarks(image, default_predictor)
            shape = shape[36:48, :] # keep only eye part!
            counter += 1
            tmp_dic = {}
            tmp_dic['original_name'] = fp
            tmp_dic['img_idx'] = counter


            right_eye, left_eye = detect_eye(image, shape)
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

    def predict_eyelid_position(self, img_info_df):
        left_data_dir = self.processed_eye_dir + '/left/'
        right_data_dir = self.processed_eye_dir + '/right/'

        os.system('mkdir -p ' + self.final_output_dir + '/left')
        os.system('mkdir -p ' + self.final_output_dir + '/right')

        measurement_results = []
        for img_idx in tqdm(img_info_df.img_idx.values):
            tmp_dict = {}
            if self.compare_with_manual_measurement:
                # load manual measurements
                pass

            # left eye
            left_predictor = dlib.shape_predictor(self.mrd_predictor_dir + '/left_eye_predictor.dat')
            left_img = cv2.imread(left_data_dir + 'left_' + str(img_idx) + '.png')
            output, (xc1, y_lower), (xc1, yc1), (xc1, y_upper), r1 = \
                measure_single_image(left_img, left_predictor)
            cv2.imwrite(self.final_output_dir + '/left/left_' + str(img_idx) + '.jpg', output)

            # left eye
            right_predictor = dlib.shape_predictor(self.mrd_predictor_dir + '/right_eye_predictor.dat')
            right_img = cv2.imread(right_data_dir + 'right_' + str(img_idx) + '.png')
            output, (xc1, y_lower), (xc1, yc1), (xc1, y_upper), r1 = \
                measure_single_image(right_img, right_predictor)
            cv2.imwrite(self.final_output_dir + '/right/right_' + str(img_idx) + '.jpg', output)
