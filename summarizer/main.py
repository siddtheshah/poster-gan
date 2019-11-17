from train import SummaryTrainer
from dataset import SummaryDataset
from network import SummaryNetwork

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import time
import json

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--storage_dir', default='/mnt/', action='store', help='Specify path of results and model data')
parser.add_argument('--image_dir', default='/mnt/movie_poster_images', action='store', help='Specify image data path')
parser.add_argument('--run_name', help='Specify a run name. (Required)', action='store')
args = parser.parse_args()

def train_new_model(configs):
    pass

def eval_model(configs):
    pass

def main():
    with open('config.json') as config_file:
        configs = json.load(config_file)
    if args.train:
        train_new_model(configs)

    return

if __name__ == "__main__":
    main()