from __future__ import division

import os
import time
import csv
from glob import glob

from dcgan.ops import *
from dcgan.utils import *

import tensorflow.compat.v1 as tf_v1
import imghdr

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=8, sample_num=32, output_height=64, output_width=64,
                 grid_height=8, grid_width=8,
                 y_dim=None, z_dim=None, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, data_dir=None,
                 input_fname_pattern='*.jpg', save_dir=None, results_dir=None, sample_rate=None,
                 nbr_of_layers_d=5, nbr_of_layers_g=5, use_checkpoints=True):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.grid_width = grid_width
        self.grid_height = grid_height

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        self.save_dir = save_dir

        self.sample_rate = sample_rate
        self.nbr_of_layers_d = nbr_of_layers_d
        self.nbr_of_layers_g = nbr_of_layers_g
        self.use_checkpoints = use_checkpoints
        self.results_dir = results_dir

        self.data = glob(os.path.join(self.data_dir, self.input_fname_pattern))
        self.c_dim = 3

        if len(self.data) < self.batch_size:
            raise Exception("[!] Entire dataset size is less than the configured batch_size")

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf_v1.placeholder(tf_v1.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf_v1.placeholder(
            tf_v1.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf_v1.placeholder(
            tf_v1.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf_v1.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf_v1.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf_v1.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf_v1.ones_like(self.D)))
        self.d_loss_fake = tf_v1.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf_v1.zeros_like(self.D_)))
        self.g_loss = tf_v1.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf_v1.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf_v1.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf_v1.train.Saver()

    def train(self, config):
        d_optim = tf_v1.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf_v1.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf_v1.global_variables_initializer().run()
        except:
            tf_v1.initialize_all_variables().run()

        self.g_sum = tf_v1.summary.merge([self.z_sum, self.d__sum,
                                          self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf_v1.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = [file for file in self.data[0:self.sample_num] if imghdr.what(file) == 'jpeg']
        sample = [
            get_image(sample_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for sample_file in sample_files]
        sample = [s for s in sample if s.shape[0] == self.input_height and s.shape[1] == self.input_width]
        all_images = []
        for s in sample:
            if len(s.shape) >= self.c_dim:
                all_images.append(np.array(s).astype(np.float32))
        sample_inputs = np.stack(all_images, 0)

        counter = 1
        start_time = time.time()
        if self.use_checkpoints:
            could_load, checkpoint_counter = self.load(self.save_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            self.data = glob(os.path.join(self.data_dir, self.input_fname_pattern))
            batch_idxs = min(len(self.data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_files = [b for b in batch_files if imghdr.what(b) == "jpeg"]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=self.crop,
                              grayscale=self.grayscale) for batch_file in batch_files]
                all_images = []
                for b in batch:
                    if len(b.shape) >= self.c_dim:
                        if b.shape[0] == self.input_height and b.shape[1] == self.input_width:
                            all_images.append(b)
                if len(all_images) < self.batch_size:
                    more_needed = self.batch_size - len(all_images)
                    if more_needed > self.batch_size/2:
                        continue # data is way too corrupt
                    all_images = all_images + all_images[:more_needed]
                batch = np.stack(all_images, 0)

                for i in range(len(batch)):
                    if len(batch[i].shape) != 3:
                        batch[i] = np.moveaxis(np.array([batch[i]] * 3), 0, 2)

                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if self.sample_rate is not None and (self.sample_rate == 1 or np.mod(counter, self.sample_rate) == 1):
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        save_images(samples, (self.grid_height, self.grid_width),
                                    '{}/train_{:02d}_{:04d}.png'.format(self.results_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if self.use_checkpoints and np.mod(counter, 500) == 2:
                    self.save(self.save_dir, counter)

                if (np.mod(counter, 30000)) == 2500:
                    self.export(os.path.join(self.save_dir, "intermediate" + str(counter)))

    def discriminator(self, image, y=None, reuse=False):
        with tf_v1.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                nbr_layers = self.nbr_of_layers_d

                print('init discriminator with ' + str(nbr_layers) + ' layers ...')
                previous_layer = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                for i in range(1, nbr_layers - 1):
                    output_dim = self.df_dim * (2 ** i)
                    layer_name = 'd_h' + str(i) + '_conv'
                    previous_layer = lrelu(
                        batch_norm(name='d_bn' + str(i))(conv2d(previous_layer, output_dim, name=layer_name)))

                layer_name = 'd_h' + str(nbr_layers - 1) + '_lin'
                last_layer = linear(tf_v1.reshape(previous_layer, [self.batch_size, -1]), 1, layer_name)
                return tf_v1.nn.sigmoid(last_layer), last_layer
            else:
                yb = tf_v1.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf_v1.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf_v1.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf_v1.variable_scope("generator") as scope:
            if not self.y_dim:
                nbr_layers = self.nbr_of_layers_g
                print('init generator with ' + str(nbr_layers) + ' layers ...')

                heights = []
                widths = []

                prev_h, prev_w = self.output_height, self.output_width
                heights.append(prev_h)
                widths.append(prev_w)

                for i in range(1, nbr_layers):
                    prev_h, prev_w = conv_out_size_same(prev_h, 2), conv_out_size_same(prev_w, 2)
                    heights.append(prev_h)
                    widths.append(prev_w)

                mul = 2 ** (nbr_layers - 2)

                height = heights[nbr_layers - 1]
                width = widths[nbr_layers - 1]
                z_ = linear(z, self.gf_dim * mul * height * width, 'g_h0_lin')
                prev_layer = tf_v1.reshape(z_, [-1, heights[nbr_layers - 1], widths[nbr_layers - 1], self.gf_dim * mul])
                prev_layer = tf_v1.nn.relu(batch_norm(name='g_bn0')(prev_layer))

                for i in range(1, nbr_layers - 1):
                    mul = mul // 2
                    height = heights[nbr_layers - 1 - i]
                    width = widths[nbr_layers - 1 - i]
                    layer_name = 'g_h' + str(i)
                    prev_layer = deconv2d(prev_layer, [self.batch_size, height, width, self.gf_dim * mul],
                                          name=layer_name)
                    prev_layer = tf_v1.nn.relu(batch_norm(name='g_bn' + str(i))(prev_layer))

                layer_name = 'g_h' + str(nbr_layers - 1)
                last_layer = deconv2d(prev_layer, [self.batch_size, heights[0], widths[0], self.c_dim], name=layer_name)

                return tf_v1.nn.tanh(last_layer)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf_v1.expand_dims(tf_v1.expand_dims(y, 1),2)
                yb = tf_v1.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf_v1.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf_v1.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf_v1.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf_v1.nn.relu(self.g_bn2(deconv2d(h1,
                                                       [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf_v1.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tf_v1.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                nbr_layers = self.nbr_of_layers_g

                heights = []
                widths = []

                prev_h, prev_w = self.output_height, self.output_width
                heights.append(prev_h)
                widths.append(prev_w)

                for i in range(1, nbr_layers):
                    prev_h, prev_w = conv_out_size_same(prev_h, 2), conv_out_size_same(prev_w, 2)
                    heights.append(prev_h)
                    widths.append(prev_w)

                mul = 2 ** (nbr_layers - 2)

                prev_layer = tf_v1.reshape(
                    linear(z, self.gf_dim * mul * heights[nbr_layers - 1] * widths[nbr_layers - 1], 'g_h0_lin'),
                    [-1, heights[nbr_layers - 1], widths[nbr_layers - 1], self.gf_dim * mul])
                prev_layer = tf_v1.nn.relu(self.g_bn0(prev_layer, train=False))

                for i in range(1, nbr_layers - 1):
                    mul = mul // 2
                    h = heights[nbr_layers - i - 1]
                    w = widths[nbr_layers - i - 1]
                    name = 'g_h' + str(i)
                    prev_layer = deconv2d(prev_layer, [self.batch_size, h, w, self.gf_dim * mul], name=name)
                    prev_layer = tf_v1.nn.relu(batch_norm(name='g_bn' + str(i))(prev_layer, train=False))

                last_layer = deconv2d(prev_layer, [self.batch_size, heights[0], widths[0], self.c_dim],
                                      name='g_h' + str(nbr_layers - 1))
                return tf_v1.nn.tanh(last_layer)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf_v1.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf_v1.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf_v1.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf_v1.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf_v1.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf_v1.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf_v1.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


    def save(self, save_dir, step):
        model_name = "DCGAN.model"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.saver.save(self.sess,
                        os.path.join(save_dir, model_name),
                        global_step=step)

    def export(self, export_dir):
        tf_v1.saved_model.simple_save(self.sess,
                                      os.path.join(export_dir, "generator"),
                                      inputs={"generator_input": self.z},
                                      outputs={"generator_output": self.G})

        tf_v1.saved_model.simple_save(self.sess,
                                      os.path.join(export_dir, "discriminator"),
                                      inputs={"discriminator_input": self.inputs},
                                      outputs={"discriminator_output": self.D_logits})

    def load(self, save_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf_v1.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(save_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
