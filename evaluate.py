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
"""Quantitative evaluation of view synthesis results.

This script compares data and dumps scores to a JSON file.
"""
import os
import json
import numpy as np
import PIL.Image as pil
import tensorflow as tf

from multiprocessing.dummy import Pool

from tensorflow import app

flags = tf.app.flags
flags.DEFINE_string('result_root', '/tmp/results',
                    'Root directory for writing results.')
flags.DEFINE_string('model_name', 'siggraph_model_20180701',
                    'Name of model to evaluate.')
flags.DEFINE_string('data_split', 'test', 'Split of the data to run on.')
flags.DEFINE_string('output_table', 'output.json',
                    'Filename for writing the output.')
FLAGS = flags.FLAGS


def load_image(imfile):
  fh = tf.gfile.GFile(imfile, 'r')
  raw_im = pil.open(fh)
  return np.array(raw_im, dtype=np.float32)


def collect_examples(result_root, model_names, data_split):
  """Find examples that exist for all models."""
  counts = {}
  for model_name in model_names:
    examples = os.listdir(os.path.join(result_root, model_name, data_split))
    for e in examples:
      counts[e] = counts.get(e, 0) + 1
  result = [k for k, v in counts.items() if v == len(model_names)]
  skipped = [k for k, v in counts.items() if v != len(model_names)]
  assert not skipped
  return result


def evaluate_one(result_root, model_name, data_split, example):
  """Compare one example on one model, returning ssim and PSNR scores."""
  example_dir = os.path.join(result_root, model_name, data_split, example)
  tgt_file = tf.gfile.Glob(example_dir + '/tgt_image_*')[0]
  tgt_image = tf.convert_to_tensor(load_image(tgt_file), dtype=tf.float32)
  pred_file = tf.gfile.Glob(example_dir + '/output_image_*')[0]
  pred_image = tf.convert_to_tensor(load_image(pred_file), dtype=tf.float32)

  ssim = tf.image.ssim(pred_image, tgt_image, max_val=255.0)
  psnr = tf.image.psnr(pred_image, tgt_image, max_val=255.0)

  with tf.Session() as sess:
    return sess.run(ssim).item(), sess.run(psnr).item()


def evaluate_example(result_root, model_names, data_split, example):
  ssims = []
  psnrs = []
  tf.logging.info('Starting %s', example)
  for model_name in model_names:
    ssim, psnr = evaluate_one(FLAGS.result_root, model_name, FLAGS.data_split,
                              example)
    ssims.append(ssim)
    psnrs.append(psnr)
  return ssims, psnrs


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  result_root = FLAGS.result_root
  data_split = FLAGS.data_split
  model_names = FLAGS.model_name.split(',')

  examples = collect_examples(result_root, model_names, data_split)
  examples.sort()
  examples = examples
  model_names = model_names

  tf.logging.info('Models: %s', model_names)
  tf.logging.info('%d examples', len(examples))

  pool = Pool(processes=20)
  all_data = pool.map(
      lambda e: evaluate_example(result_root, model_names, data_split, e),
      examples)
  output_dir = 'eval_out'
  data = {
      'model_names': model_names,
      'examples': examples,
      'ssim': [ssim for (ssim, psnr) in all_data],
      'psnr': [psnr for (ssim, psnr) in all_data],
  }
  with open(FLAGS.output_table, 'w') as f:
    json.dump(data, f)
  tf.logging.info('Output written to %s' % FLAGS.output_table)


if __name__ == '__main__':
  app.run()
