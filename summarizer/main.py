import summarizer.eval
from summarizer.train import SummaryTrainer
from summarizer.dataset import SummaryDataset
from summarizer.network import SummaryNetwork

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
parser.add_argument('--trailer_dir', default='/mnt/movie_trailer_frames', action='store', help='Specify poster data path')
parser.add_argument('--image_dir', default='/mnt/movie_poster_images', action='store', help='Specify poster data path')
parser.add_argument('--run_name', help='Specify a run name. (Required)', action='store')
args = parser.parse_args()

def train_new_model(configs):
    # Compose the training step
    generator = None # Use TF.load_model to load the gan as a module of some kind.
    discriminator = None # Again, use TF.load_model to load the discriminator side.
    model = SummaryNetwork(generator, configs["weight_decay"])
    dataset = SummaryDataset(args.trailer_dir, args.poster_dir, configs["validation_folds"])
    optimizer = tf_v1.keras.optimizers.Adam()
    save_dir = os.path.join(args.storage_dir, args.run_name, "model")
    loss = summarizer.eval.grad
    alpha = configs["alpha"]
    beta = configs["beta"]
    batch_size = configs["batch_size"]
    epochs_per_validation = configs["epochs_per_validation"]
    # Create the trainer and train.
    trainer = SummaryTrainer(model, discriminator, dataset, save_dir, optimizer, loss, alpha, beta, batch_size, epochs_per_validation)
    trainer.train(configs["epochs"])
    return

def eval_model(configs):
    pass

def main():
    with open('config.json') as config_file:
        configs = json.load(config_file)
    if args.train:
        train_new_model(configs)

    # if args.eval:
        # do advanced plotting code here.


    return

if __name__ == "__main__":
    main()