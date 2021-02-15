import sys, getopt
import pandas as pd
import json
from os import listdir
import pickle

sys.path.insert(1, './src/')
from trainer import *
from utils import *

def main():

    if len(sys.argv) == 1:
        target = 'all'
        config = json.load(open('config/8_points.json', 'r'))
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        config = json.load(open('config/8_points.json', 'r'))
    else:
        target = sys.argv[1]
        config = json.load(open('config/' + sys.argv[2], 'r'))

    T = Trainer(config)

    if target == 'crop_img':
        print()
        img_info_df = T.process_full_face_img()
        img_info_df['to_predict'] = True
        img_info_df.to_csv(T.curr_dir + 'data/processed/' + T.data_identifier + '/img_info_df.csv', index = False)
        print(' => Done image cropping; got', img_info_df.shape[0], 'images; saved at:', \
                T.curr_dir + 'data/processed/' + T.data_identifier + '/img_info_df.csv')
        print()
    if target == 'run_measurement':
        print()
        img_info_df = pd.read_csv(T.curr_dir + 'data/processed/' + T.data_identifier + '/img_info_df.csv')\
                        .query('to_predict == True').reset_index(drop = True)
        print(' => Loaded image info; found', img_info_df.shape[0], 'valid images to classify.')
        print()
        T.predict_eyelid_position(img_info_df)



if __name__ == "__main__":
    main()
