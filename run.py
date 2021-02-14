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
        T.process_full_face_img()

if __name__ == "__main__":
    main()
