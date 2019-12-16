from dcgan.model import DCGAN
from dcgan.utils import pp, visualize, show_all_variables, visualize2

import io
import os
import os.path
from os import listdir
from os.path import isfile, join

from PIL import Image

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import numpy as np

import json
import argparse

flags = tf_v1.app.flags
flags.DEFINE_boolean("train", False, "Train a new model")
flags.DEFINE_boolean("eval", False, "Run eval on a model")
flags.DEFINE_string("run_name", None, "Run name (Required)")
flags.DEFINE_integer("epoch", 40, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", None, "The size of batch images [64]")
flags.DEFINE_integer("grid_height", 8, "Grid Height")
flags.DEFINE_integer("grid_width", 8, "Grid Width")
flags.DEFINE_integer("input_height", None, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", None, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_integer("sample_rate", None, "If == 5, it will take a sample image every 5 iterations")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("use_checkpoints", True, "Save and load checkpoints")
flags.DEFINE_integer("z_dim", 1024, "Number of images to generate during test. [10]")
FLAGS = flags.FLAGS

# tf_v1.disable_eager_execution()

if FLAGS.run_name is None:
    raise Exception('Run name is required!')
# default batch_size
if FLAGS.batch_size is None and FLAGS.grid_height is not None and FLAGS.grid_width is not None:
    batch_size = FLAGS.grid_height * FLAGS.grid_width
elif FLAGS.batch_size is not None:
    batch_size = FLAGS.batch_size
else:
    raise Exception('grid_height/grid_width or batch_size must be provided')

# default size parameters
input_width = FLAGS.input_width
input_height = FLAGS.input_height
output_width = FLAGS.output_width
output_height = FLAGS.output_height


with open('config.json') as config_file:
    configs = json.load(config_file)

data_dir = configs["poster_dir"]
run_dir = os.path.join(configs["gan_path"], FLAGS.run_name)
save_dir = os.path.join(run_dir, "checkpoints")
export_dir = os.path.join(run_dir, "export")
results_dir = os.path.join(run_dir, "results")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if (input_height is None and input_width is None) or (output_height is None and output_width is None):
    first_image = [f for f in listdir(data_dir) if isfile(join(data_dir, f))][0]
    image_data = open(os.path.join(data_dir, first_image), "rb").read()
    image = Image.open(io.BytesIO(image_data))
    rgb_im = image.convert('RGB')
    input_width = rgb_im.size[0]
    output_width = rgb_im.size[0]
    input_height = rgb_im.size[1]
    output_height = rgb_im.size[1]


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if FLAGS.use_checkpoints and not os.path.exists(save_dir):
        os.makedirs(FLAGS.save_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    gpu_options = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.95)
    run_config = tf_v1.ConfigProto()
    run_config.gpu_options.allow_growth = True

    if FLAGS.train:
        with tf_v1.Session(config=run_config) as sess:
            dcgan = DCGAN(
                sess,
                input_width=input_width,
                input_height=input_height,
                output_width=output_width,
                output_height=output_height,
                grid_height=FLAGS.grid_height,
                grid_width=FLAGS.grid_width,
                batch_size=batch_size,
                sample_num=batch_size,
                data_dir=data_dir,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                save_dir=save_dir,
                results_dir=results_dir,
                sample_rate=FLAGS.sample_rate,
                z_dim=FLAGS.z_dim,
                nbr_of_layers_d=5,
                nbr_of_layers_g=5,
                use_checkpoints=FLAGS.use_checkpoints)

            show_all_variables()

            dcgan.train(FLAGS)
            dcgan.export(export_dir)

    if FLAGS.eval:
        generator_path = os.path.join(export_dir, "generator")
        if not os.path.exists(generator_path):
            raise Exception("[!] Train a model first, then run test mode")

        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization
        OPTION = 0

        visualize2(generator_path, FLAGS, batch_size, results_dir)


if __name__ == '__main__':
    tf_v1.app.run()
