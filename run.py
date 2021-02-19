import sys, getopt
import pandas as pd
import json
from os import listdir
import pickle

sys.path.insert(1, './src/')
from trainer import *
from utils import *


def parse_arg(argv):
    try:
        opts, args = getopt.getopt(argv, "hm:c", ["help", "model=", "crop"])
    except getopt.GetoptError:
        display_help_menu()
        sys.exit(2)
    # print(opts, args)
    target = None
    model = None
    # testing = False

    for opt, arg in opts:
        if opt == '-h':
            display_help_menu()
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-c", "--crop"):
            target = "crop"
    return target, model


def main():
    target, model = parse_arg(sys.argv[1:])
    config_file_name = model + '.json'
    if config_file_name not in listdir('config'):
        print('=> [run.py: main()] No json file with name:"' + config_file_name + '" found! terminating...')
        return

    # Load json file
    with open('config/' + config_file_name) as json_file:
        config = json.load(json_file)
    # if len(sys.argv) == 1:
    #     target = 'all'
    #     config = json.load(open('config/8_points.json', 'r'))
    # elif len(sys.argv) == 2:
    #     target = sys.argv[1]
    #     config = json.load(open('config/8_points.json', 'r'))
    # else:
    #     target = sys.argv[1]
    #     config = json.load(open('config/' + sys.argv[2], 'r'))

    T = Trainer(config)

    if target == 'crop':
        print()
        img_info_df = T.process_full_face_img()
        img_info_df['to_predict'] = True
        meta_data_dir = T.curr_dir + 'data/meta_data/' + T.model_name + '.csv'
        img_info_df.to_csv(meta_data_dir, index = False)
        print(' => Done image cropping; got', img_info_df.shape[0], 'images; saved at: data/processed/' + T.model_name)
        print()
    elif target == 'run_measurement':
        print()
        img_info_df = pd.read_csv(T.curr_dir + 'data/meta_data/' + T.model_name + '/img_info_df.csv')\
                        .query('to_predict == True').reset_index(drop = True)
        print(' => Loaded image info; found', img_info_df.shape[0], 'valid images to classify.')
        print()
        T.predict_eyelid_position(img_info_df)
    elif target == 'all':
        pass



if __name__ == "__main__":
    main()
