# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class that computes the Wasserstein distance in tensorflow.

   The implementation follows Algorithm 2 in [Genevay Aude, Marco Cuturi,
   Gabriel Peyre, Francis Bach, "Stochastic Optimization for Large-scale
   Optimal Transport", NIPS 2016], which compares a distribution to a
   fixed set of samples. Internally, base distances are recomputed a lot.
   To just compute the Wasserstein distance between to sets of points,
   don't use this code, just do a bipartitle matching.
"""

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v1 as tf_v2

tf_v1.enable_eager_execution()

class Wasserstein(object):
  """Class to hold (ref to) data and compute Wasserstein distance."""

  def __init__(self, batch_size, source, target, basedist=None):
    """Inits Wasserstein with source and target data."""
    self.source = source
    self.target = target
    self.gradbs = batch_size  # number of source sample to compute gradient
    if basedist is None:
      basedist = self.l2dist
    self.basedist = basedist

  def add_summary_montage(self, images, name, num=9):
    vis_images = tf_v1.split(images[:num], num_or_size_splits=num, axis=0)
    vis_images = tf_v1.concat(vis_images, axis=2)
    tf_v1.summary.image(name, vis_images)
    return vis_images

  def add_summary_images(self, num=9):
    """Visualize source images and nearest neighbors from target."""
    vis_images = self.add_summary_montage(self.source, 'source_ims', num)

    _ = self.add_summary_montage(self.target, 'target_ims', num)

    c_xy = self.basedist(self.source, self.target)  # pairwise cost
    idx = tf_v1.argmin(c_xy, axis=1)  # find nearest neighbors
    matches = tf_v1.gather(self.target, idx)
    vis_matches = self.add_summary_montage(matches, 'neighbors_ims', num)

    vis_both = tf_v1.concat([vis_images, vis_matches], axis=1)
    tf_v1.summary.image('matches_ims', vis_both)

    return

  def l2dist(self, source, target):
    """Computes pairwise Euclidean distances in tensorflow."""
    def flatten_batch(x):
      dim = tf_v1.reduce_prod(tf_v1.shape(x)[1:])
      return tf_v1.reshape(x, [-1, dim])
    def scale_batch(x):
      dim = tf_v1.reduce_prod(tf_v1.shape(x)[1:])
      return x/tf_v1.sqrt(tf_v1.cast(dim, tf_v1.float32))
    def prepare_batch(x):
      return scale_batch(flatten_batch(x))

    target_flat = prepare_batch(target)  # shape: [bs, nt]
    target_sqnorms = tf_v1.reduce_sum(tf_v1.square(target_flat), axis=1, keep_dims=True)
    target_sqnorms_t = tf_v1.transpose(target_sqnorms)

    source_flat = prepare_batch(source)  # shape: [bs, ns]
    source_sqnorms = tf_v1.reduce_sum(tf_v1.square(source_flat), axis=1, keep_dims=True)

    dotprod = tf_v1.matmul(source_flat, target_flat, transpose_b=True)  # [ns, nt]
    sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t
    dist = tf_v1.sqrt(tf_v1.nn.relu(sqdist))  # potential tiny negatives are suppressed
    return dist  # shape: [ns, nt]

  def grad_hbar(self, v, gradbs, reuse=True):
    """Compute gradient of hbar function for Wasserstein iteration."""

    c_xy = self.basedist(self.source, self.target)
    c_xy -= v  # [gradbs, trnsize]
    idx = tf_v1.argmin(c_xy, axis=1)               # [1] (index of subgradient)
    xi_ij = tf_v1.one_hot(idx, self.gradbs)  # find matches, [gradbs, trnsize]
    xi_ij = tf_v1.reduce_mean(xi_ij, axis=0, keep_dims=True)    # [1, trnsize]
    grad = 1./self.gradbs - xi_ij  # output: [1, trnsize]
    return grad

  def hbar(self, v, reuse=True):
    """Compute value of hbar function for Wasserstein iteration."""

    c_xy = self.basedist(self.source, self.target)
    c_avg = tf_v1.reduce_mean(c_xy)
    c_xy -= c_avg
    c_xy -= v

    c_xy_min = tf_v1.reduce_min(c_xy, axis=1)  # min_y[ c(x, y) - v(y) ]
    c_xy_min = tf_v1.reduce_mean(c_xy_min)     # expectation wrt x
    return tf_v1.reduce_mean(v, axis=1) + c_xy_min + c_avg # avg wrt y

  def k_step(self, k, v, vt, c, reuse=True):
    """Perform one update step of Wasserstein computation."""
    grad_h = self.grad_hbar(vt, gradbs=self.gradbs, reuse=reuse)
    vt = vt + c/tf_v1.sqrt(k)*grad_h
    v = ((k-1.)*v + vt)/k
    return k+1, v, vt, c

  def dist(self, C=.1, nsteps=10, reset=False):
    """Compute Wasserstein distance (Alg.2 in [Genevay etal, NIPS'16])."""
    vtilde = tf_v1.Variable(tf_v1.zeros([1, self.gradbs]), name='vtilde')
    v = tf_v1.Variable(tf_v1.zeros([1, self.gradbs]), name='v')
    k = tf_v1.Variable(1., name='k')

    k = k.assign(1.)  # restart averaging from 1 in each call
    if reset:  # used for randomly sampled target data, otherwise warmstart
      v = v.assign(tf_v1.zeros([1, self.gradbs]))  # reset every time graph is evaluated
      vtilde = vtilde.assign(tf_v1.zeros([1, self.gradbs]))

    # (unrolled) optimization loop. first iteration, create variables
    k, v, vtilde, C = self.k_step(k, v, vtilde, C, reuse=False)
    # (unrolled) optimization loop. other iterations, reuse variables
    k, v, vtilde, C = tf_v1.while_loop(cond=lambda k, *_: k < nsteps,
                                            body=self.k_step,
                                            loop_vars=[k, v, vtilde, C])
    v = tf_v1.stop_gradient(v)  # only transmit gradient through cost
    val = self.hbar(v)
    return tf_v1.reduce_mean(val)



