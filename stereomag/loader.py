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
"""Loads video sequence data using the tf Dataset API.

For easiest setup, use create_from_flags, then you won't have to specify any
additional options.
"""
import os.path
import tensorflow as tf
from tensorflow import flags
import datasets

flags.DEFINE_integer('epochs', -1,
                     'Epochs of training data, or -1 to continue indefinitely.')

flags.DEFINE_integer('image_height', 256, 'Image height in pixels.')
flags.DEFINE_integer('image_width', 256, 'Image width in pixels.')

flags.DEFINE_integer('sequence_length', 10, 'Sequence length for each example.')
flags.DEFINE_integer('min_stride', 3, 'Minimum stride for sequence.')
flags.DEFINE_integer('max_stride', 10, 'Maximum stride for sequence.')

flags.DEFINE_float('augment_min_scale', 1.0,
                   'Minimum scale for data augmentation.')
flags.DEFINE_float('augment_max_scale', 1.15,
                   'Maximum scale for data augmentation.')

flags.DEFINE_integer('batch_size', 8, 'The size of a sample batch.')

FLAGS = flags.FLAGS

# The lambdas in this file are only long because of clear argument names.
# pylint: disable=g-long-lambda


class Loader(object):
  """Process video sequences into a dataset for use in training or testing."""

  def __init__(
      self,
      # What to load
      all_sequences,
      image_dir,
      # Whether data to be loaded is for use in training (vs. test). This
      # affects whether data is sampled pseudorandomly (in the case of training
      # data) or deteministically (in the case of validation/test data), and
      # whether data augmentation is applied (training data only).
      training=True,
      # Repetition (for training dataset only)
      epochs=-1,
      # Output dimensions
      image_height=None,
      image_width=None,
      # Choosing subsequences
      sequence_length=None,
      min_stride=1,
      max_stride=1,
      # Custom processing
      map_function=None,
      # Augmentation
      min_scale=1.0,
      max_scale=1.0,
      # Batching
      batch_size=1,
      # Tuning for efficiency
      parallelism=10,
      parallel_image_reads=50,
      prefetch=1):
    """Produce tensorflow datasets of sampled ViewSequences.

    When loading training data (training == True), the data is randomly
    shuffled, uses random subsequences, and repeats indefinitely. Otherwise, the
    datasets are not shuffled and the code samples the data deterministically,
    always selecting subsequences from the middle of a sequences using a stride
    of (min_stride + max_stride)//2. All data is batched according to
    batch_size. For validation/test sets (training == False), if the size is not
    an exact multiple of batch_size, the last few elements are discarded so that
    only full batches are present in the batched dataset.

    Args:
      all_sequences: ViewSequences dataset, without images
      image_dir: the base path for images
      epochs: (int) How many epochs of training data to generate, or -1 to
        continue indefinitely.
      image_height: (int) Images are resized to this height
      image_width: (int) Images are resized to this width
      sequence_length: (int) A subsequence of this length will be chosen from
        each sequence read in
      min_stride: (int) Minimum stride between elements of subsequence
      max_stride: (int) Maximum stride between elements of subsequence
      training: (bool) Whether the data loaded is intended for training
      map_function: function to apply to sequences before batching, or None
      min_scale: (float) Minimum scale for data augmentation (training only)
      max_scale: (float) Maximum scale for data augmentation (training only)
      batch_size: (int) Batch size
      parallelism: (int) How many sequences to process in parallel
      parallel_image_reads: (int) How many images to read in parallel
      prefetch: (int) How many items to prefetch from each dataset
    """

    def prepare_for_training(sequences):
      """Steps applied to training dataset only."""
      # Random shuffling, random subsequences and random reversal for training.
      # Also we make it repeat indefinitely.
      shuffled = sequences.shuffle(1000).repeat(epochs)
      # Discard sequences that are too short to generate a subsequence at
      # max stride.
      required_length = (sequence_length - 1) * max_stride + 1
      filtered = shuffled.filter(
          lambda sequence: tf.greater_equal(sequence.length(), required_length))
      subsequences = filtered.map(
          lambda sequence: sequence.random_subsequence(sequence_length, min_stride, max_stride)
      )
      return subsequences.map(lambda sequence: sequence.random_reverse())

    def prepare_for_testing(sequences):
      """Steps applied to validation and test datasets only."""
      # No shuffling, deterministic subsequences and reversal for testing.
      stride = (min_stride + max_stride) // 2
      # Discard sequences that are too short to generate a subsequence at
      # stride.
      required_length = (sequence_length - 1) * stride + 1
      filtered = sequences.filter(
          lambda sequence: tf.greater_equal(sequence.length(), required_length))
      subsequences = filtered.map(
          lambda sequence: sequence.central_subsequence(sequence_length, stride)
      )
      return subsequences.map(lambda sequence: sequence.deterministic_reverse())

    def load_image_data(sequences):
      return sequences.map(
          datasets.load_image_data(image_dir, image_height, image_width,
                                   parallel_image_reads),
          num_parallel_calls=parallelism)

    def full_batches(sequence):
      return tf.equal(tf.shape(sequence.id)[0], batch_size)

    def set_batched_shape(sequence):
      return sequence.set_batched_shape(batch_size, sequence_length)

    def batch_and_prefetch(dataset):
      return (dataset.padded_batch(batch_size, dataset.output_shapes)
              .filter(full_batches).map(set_batched_shape).prefetch(prefetch))

    # Training sequences are treated different because they are randomised
    # whereas validation and test should be deterministic to ensure consistent
    # results.
    if training:
      sequences = prepare_for_training(all_sequences)
    else:
      sequences = prepare_for_testing(all_sequences)

    # Load images
    sequences_with_images = load_image_data(sequences)

    # Data augmentation
    if training and (min_scale != 1.0 or max_scale != 1.0):
      sequences_with_images = sequences_with_images.map(
          lambda sequence: sequence.random_scale_and_crop(
              min_scale, max_scale, image_height, image_width))

    # Custom processing
    if map_function:
      sequences_with_images = sequences_with_images.map(
          map_function, num_parallel_calls=parallelism)

    # Batching. We discard batches that are smaller than the batch size,
    # because it's easier to write downstream code knowing what the shapes
    # will be.
    sequences = batch_and_prefetch(sequences_with_images)

    # Things we expose to the calling code
    self.sequences = sequences


# Create a dataset configured with the flags specified at the top of this file.
def create_from_flags(cameras_glob='train/????????????????.txt',
                      image_dir='images',
                      training=True,
                      map_function=None):
  """Convenience function to return a Loader configured by flags."""
  assert tf.gfile.IsDirectory(image_dir)  # Ensure the provided path is valid.
  assert tf.gfile.ListDirectory(image_dir) > 0  # Ensure that some data exists.
  parallelism = 10

  assert tf.gfile.Glob(cameras_glob)
  files = tf.data.Dataset.list_files(cameras_glob, False)
  lines = files.map(datasets.read_file_lines, num_parallel_calls=parallelism)
  sequences = lines.map(
      datasets.parse_camera_lines, num_parallel_calls=parallelism)

  return Loader(
      sequences,
      image_dir,
      training,
      epochs=FLAGS.epochs,
      # Output dimensions
      image_height=FLAGS.image_height,
      image_width=FLAGS.image_width,
      # Choosing subsequencues
      sequence_length=FLAGS.sequence_length,
      min_stride=FLAGS.min_stride,
      max_stride=FLAGS.max_stride,
      # Custom processing
      map_function=map_function,
      # Augmentation
      min_scale=FLAGS.augment_min_scale,
      max_scale=FLAGS.augment_max_scale,
      # Batching
      batch_size=FLAGS.batch_size,
      # Tuning for efficiency
      parallelism=parallelism,
      parallel_image_reads=50,
      prefetch=1)
