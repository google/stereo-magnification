#!/usr/bin/python
#
# Copyright 2018 Google LLC
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
"""Main script for training multiplane image (MPI) network.
"""
from __future__ import division
import tensorflow as tf

from stereomag.sequence_data_loader import SequenceDataLoader
from stereomag.mpi import MPI

# Note that the flags below are a subset of all flags. The remainder (data
# loading relevant) are defined in loader.py.
flags = tf.app.flags
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Location to save the models.')
flags.DEFINE_string('cameras_glob', 'train/????????????????.txt',
                    'Glob string for training set camera files.')
flags.DEFINE_string('image_dir', 'images',
                    'Path to training image directories.')
flags.DEFINE_string('experiment_name', '', 'Name for the experiment to run.')
flags.DEFINE_integer('random_seed', 8964, 'Random seed.')
flags.DEFINE_string(
    'which_loss', 'pixel', 'Which loss to use to compare '
    'rendered and ground truth images. '
    'Can be "pixel" or "VGG".')
flags.DEFINE_string('which_color_pred', 'bg',
                    'Color output format: [alpha_only,single,bg,fgbg,all].')
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1 hyperparameter for Adam optimizer.')
flags.DEFINE_integer('max_steps', 10000000, 'Maximum number of training steps.')
flags.DEFINE_integer('summary_freq', 50, 'Logging frequency.')
flags.DEFINE_integer(
    'save_latest_freq', 2000, 'Frequency with which to save the model '
    '(overwrites previous model).')
flags.DEFINE_boolean('continue_train', False,
                     'Continue training from previous checkpoint.')
flags.DEFINE_integer('num_source', 2, 'Number of source images.')
flags.DEFINE_integer(
    'shuffle_seq_length', 10,
    'Length of sequences to be sampled from each video clip. '
    'Each sequence is shuffled, and then the first '
    'num_source + 1 images from the shuffled sequence are '
    'selected as a training instance. Increasing this number '
    'results in more varied baselines in training data.')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scene depth.')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for plane sweep '
                     'volume (PSV).')
flags.DEFINE_integer('num_mpi_planes', 32, 'Number of MPI planes to predict.')
flags.DEFINE_string(
    'vgg_model_file', 'imagenet-vgg-verydeep-19.mat',
    'Location of vgg model file used to compute perceptual '
    '(VGG) loss.')
FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.random_seed)
  FLAGS.checkpoint_dir += '/%s/' % FLAGS.experiment_name
  if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  # Set up data loader
  data_loader = SequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, True,
                                   FLAGS.num_source, FLAGS.shuffle_seq_length,
                                   FLAGS.random_seed)

  train_batch = data_loader.sample_batch()
  model = MPI()
  train_op = model.build_train_graph(
      train_batch, FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_psv_planes,
      FLAGS.num_mpi_planes, FLAGS.which_color_pred, FLAGS.which_loss,
      FLAGS.learning_rate, FLAGS.beta1, FLAGS.vgg_model_file)
  model.train(train_op, FLAGS.checkpoint_dir, FLAGS.continue_train,
              FLAGS.summary_freq, FLAGS.save_latest_freq, FLAGS.max_steps)


if __name__ == '__main__':
  tf.app.run()
