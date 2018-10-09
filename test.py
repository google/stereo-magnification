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

"""Main script for evaluating multiplane image (MPI) network on a test set.
"""
from __future__ import division
import os
import tensorflow as tf

from stereomag.mpi import MPI
from stereomag.sequence_data_loader import SequenceDataLoader
from stereomag.utils import write_image
from stereomag.utils import write_intrinsics
from stereomag.utils import write_pose

# Note that the flags below are a subset of all flags. The remainder (data
# loading relevant) are defined in loader.py.
flags = tf.app.flags
flags.DEFINE_string('model_root', 'models',
                    'Root directory for model checkpoints.')
flags.DEFINE_string('model_name', 'siggraph_model_20180701',
                    'Name of the model to use for inference.')
flags.DEFINE_string('data_split', 'test',
                    'Which split to run ("train" or "test").')
flags.DEFINE_integer('num_runs', 20, 'number of runs')
flags.DEFINE_string('cameras_glob', 'test/????????????????.txt',
                    'Glob string for test set camera files.')
flags.DEFINE_string('image_dir', 'images', 'Path to test image directories.')
flags.DEFINE_integer('random_seed', 8964, 'Random seed')
flags.DEFINE_string('output_root', '/tmp/results',
                    'Root of directory to write results.')
flags.DEFINE_integer('num_source', 2, 'Number of source images.')
flags.DEFINE_integer(
    'shuffle_seq_length', 10,
    'Length of sequences to be sampled from each video clip. '
    'Each sequence is shuffled, and then the first '
    'num_source + 1 images from the shuffled sequence are '
    'selected as a test instance. Increasing this number '
    'results in more varied baselines in training data.')
flags.DEFINE_string('which_color_pred', 'bg',
                    'Color output format: [alpha_only,single,bg,fgbg,all].')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scenen depth.')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for plane sweep '
                     'volume (PSV).')
flags.DEFINE_integer('num_mpi_planes', 32, 'Number of MPI planes to predict.')
flags.DEFINE_string(
    'test_outputs', 'rgba_layers_src_images_tgt_image',
    'Which outputs to save. Can concat the following with "_": '
    '[src_images, ref_image, tgt_image, psv, fgbg, poses,'
    ' intrinsics, blend_weights, rgba_layers].')
FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.batch_size == 1, 'Currently, batch_size must be 1 when testing.'

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.reset_default_graph()
  tf.set_random_seed(FLAGS.random_seed)

  # Set up data loader.
  data_loader = SequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, False,
                                   FLAGS.num_source, FLAGS.shuffle_seq_length,
                                   FLAGS.random_seed)

  inputs = data_loader.sample_batch()
  model = MPI()
  psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                                FLAGS.num_psv_planes)
  mpi_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                                FLAGS.num_mpi_planes)
  outputs = model.infer_mpi(
      inputs['src_images'], inputs['ref_image'], inputs['ref_pose'],
      inputs['src_poses'], inputs['intrinsics'], FLAGS.which_color_pred,
      FLAGS.num_mpi_planes, psv_planes, FLAGS.test_outputs)

  if 'tgt_image' in FLAGS.test_outputs:
    rel_pose = tf.matmul(inputs['tgt_pose'],
                         tf.matrix_inverse(inputs['ref_pose']))
    outputs['output_image'] = model.mpi_render_view(
        outputs['rgba_layers'], rel_pose, mpi_planes, inputs['intrinsics'])
    outputs['output_image'] = model.deprocess_image(outputs['output_image'])

  saver = tf.train.Saver([var for var in tf.model_variables()])
  ckpt_dir = os.path.join(FLAGS.model_root, FLAGS.model_name)
  ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
  sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
  config = tf.ConfigProto()

  with sv.managed_session(config=config) as sess:
    saver.restore(sess, ckpt_file)
    for run in range(FLAGS.num_runs):
      tf.logging.info('Progress: %d/%d' % (run, FLAGS.num_runs))
      ins, outs = sess.run([inputs, outputs])

      # Output directory name: [scene]_[1st src file]_[2nd src file]_[tgt file].
      dirname = ins['ref_name'][0].split('/')[0]
      for i in range(FLAGS.num_source):
        dirname += '_%s' % (
            os.path.basename(
                ins['src_timestamps'][0][i]).split('.')[0].split('_')[-1])
      dirname += '_%s' % (
          os.path.basename(
              ins['tgt_timestamp'][0]).split('.')[0].split('_')[-1])

      output_dir = os.path.join(FLAGS.output_root, FLAGS.model_name,
                                FLAGS.data_split, dirname)
      if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

      # Write results to disk.
      if 'intrinsics' in FLAGS.test_outputs:
        with open(output_dir + '/intrinsics.txt', 'w') as fh:
          write_intrinsics(fh, ins['intrinsics'][0])

      if 'src_images' in FLAGS.test_outputs:
        for i in range(FLAGS.num_source):
          timestamp = ins['src_timestamps'][0][i]
          write_image(output_dir + '/src_image_%d_%s.png' % (i, timestamp),
                      ins['src_images'][0, :, :, i * 3:(i + 1) * 3] * 255.0)
          if 'poses' in FLAGS.test_outputs:
            write_pose(output_dir + '/src_pose_%d.txt' % i,
                       ins['src_poses'][0, i])

      if 'tgt_image' in FLAGS.test_outputs:
        timestamp = ins['tgt_timestamp'][0]
        write_image(output_dir + '/tgt_image_%s.png' % timestamp,
                    ins['tgt_image'][0] * 255.0)
        write_image(output_dir + '/output_image_%s.png' % timestamp,
                    outs['output_image'][0])
        if 'poses' in FLAGS.test_outputs:
          write_pose(output_dir + '/tgt_pose.txt', ins['tgt_pose'][0])

      if 'fgbg' in FLAGS.test_outputs:
        write_image(output_dir + '/foreground_color.png', outs['fg_image'][0])
        write_image(output_dir + '/background_color.png', outs['bg_image'][0])

      if 'blend_weights' in FLAGS.test_outputs:
        for i in range(FLAGS.num_mpi_planes):
          weight_img = outs['blend_weights'][0, :, :, i] * 255.0
          write_image(output_dir + '/foreground_weight_plane_%.3d.png' % i,
                      weight_img)

      if 'ref_image' in FLAGS.test_outputs:
        fname = os.path.basename(ins['ref_name'][0])
        write_image(output_dir + '/ref_image_%s' % fname, ins['ref_image'][0])
        write_pose(output_dir + '/ref_pose.txt', ins['ref_pose'][0])

      if 'psv' in FLAGS.test_outputs:
        for j in range(FLAGS.num_psv_planes):
          plane_img = (outs['psv'][0, :, :, j * 3:(j + 1) * 3] + 1.) / 2. * 255
          write_image(output_dir + '/psv_plane_%.3d.png' % j, plane_img)

      if 'rgba_layers' in FLAGS.test_outputs:
        for i in range(FLAGS.num_mpi_planes):
          alpha_img = outs['rgba_layers'][0, :, :, i, 3] * 255.0
          rgb_img = (outs['rgba_layers'][0, :, :, i, :3] + 1.) / 2. * 255
          write_image(output_dir + '/mpi_alpha_%.2d.png' % i, alpha_img)
          write_image(output_dir + '/mpi_rgb_%.2d.png' % i, rgb_img)


if __name__ == '__main__':
  tf.app.run()
