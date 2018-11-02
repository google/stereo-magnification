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
"""Script to generate a multiplane image (MPI) from an image pair related by a
3D translation."""

from __future__ import division
import os
import sys
import tensorflow as tf

from stereomag.mpi import MPI
from stereomag.utils import build_matrix
from stereomag.utils import write_image
from stereomag.utils import write_intrinsics
from stereomag.utils import write_pose

flags = tf.app.flags

# Input flags
flags.DEFINE_string('image1', '', 'First (reference) input image filename.')
flags.DEFINE_string('image2', '', 'Second input image filename.')
flags.DEFINE_float('xoffset', 0.0,
                   'Camera x-offset from first to second image.')
flags.DEFINE_float('yoffset', 0.0,
                   'Camera y-offset from first to second image.')
flags.DEFINE_float('zoffset', 0.0,
                   'Camera z-offset from first to second image.')
flags.DEFINE_float('fx', 0.5, 'Focal length as a fraction of image width.')
flags.DEFINE_float('fy', 0.5, 'Focal length as a fraction of image height.')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scene depth.')
flags.DEFINE_integer(
    'xshift', 0, 'Horizontal pixel shift for image2 '
    '(i.e., difference in x-coordinate of principal point '
    'from image2 to image1).')
flags.DEFINE_integer(
    'yshift', 0, 'Vertical pixel shift for image2 '
    '(i.e., difference in y-coordinate of principal point '
    'from image2 to image1).')
flags.DEFINE_string('pose1', '',
                    ('Camera pose for first image (if not identity).'
                     ' Twelve space- or comma-separated floats, forming a 3x4'
                     ' matrix in row-major order.'))
flags.DEFINE_string('pose2', '',
                    ('Pose for second image (if not identity).'
                     ' Twelve space- or comma-separated floats, forming a 3x4'
                     ' matrix in row-major order. If pose2 is specified, then'
                     ' xoffset/yoffset/zoffset flags will be used for rendering'
                     ' output views only.'))
# Output flags
flags.DEFINE_string('output_dir', '/tmp/', 'Directory to write MPI output.')
flags.DEFINE_string('test_outputs', 'rgba_layers,src_images',
                    ('Which outputs to save. Can concat the following with _'
                     ' [src_images, ref_image, psv, fgbg, poses,'
                     ' intrinsics, blend_weights, rgba_layers]'))

# Rendering images.
flags.DEFINE_boolean('render', False,
                     'Render output images at multiples of input offset.')
flags.DEFINE_string(
    'render_multiples',
    '-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12',
    'Multiples of input offset to render outputs at.')

# Model flags. Defaults are the model described in the SIGGRAPH 2018 paper.  See
# README for more details.
flags.DEFINE_string('model_root', 'models/',
                    'Root directory for model checkpoints.')
flags.DEFINE_string('model_name', 'siggraph_model_20180701',
                    'Name of the model to use for inference.')
flags.DEFINE_string('which_color_pred', 'bg',
                    'Color output format: [alpha_only,single,bg,fgbg,all].')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for PSV.')
flags.DEFINE_integer('num_mpi_planes', 32, 'Number of MPI planes to infer.')

FLAGS = flags.FLAGS


def shift_image(image, x, y):
  """Shift an image x pixels right and y pixels down, filling with zeros."""
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  x = int(round(x))
  y = int(round(y))
  dtype = image.dtype
  if x > 0:
    image = tf.concat(
        [tf.zeros([height, x, 3], dtype=dtype), image[:, :(width - x)]], axis=1)
  elif x < 0:
    image = tf.concat(
        [image[:, -x:], tf.zeros([height, -x, 3], dtype=dtype)], axis=1)
  if y > 0:
    image = tf.concat(
        [tf.zeros([y, width, 3], dtype=dtype), image[:(height - y), :]], axis=0)
  elif y < 0:
    image = tf.concat(
        [image[-y:, :], tf.zeros([-y, width, 3], dtype=dtype)], axis=0)
  return image


def crop_to_multiple(image, size):
  """Crop image to a multiple of size in height and width."""
  # Compute how much we need to remove.
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_width = width - (width % size)
  new_height = height - (height % size)
  # Crop amounts. Extra pixel goes on the left side.
  left = (width % size) // 2
  right = new_width + left
  top = (height % size) // 2
  bottom = new_height + top
  return image[top:bottom, left:right]


def crop_to_size(image, width, height):
  """Crop image to specified size."""
  shape = tf.shape(image)
  # crop_to_multiple puts the extra pixel on the left, so here
  # we make sure to remove the extra pixel from the left.
  left = (shape[1] - width + 1) // 2
  top = (shape[0] - height + 1) // 2
  right = left + width
  bottom = top + height
  return image[top:bottom, left:right]


def load_image(f, padx, pady, xshift, yshift):
  """Load an image, pad, and shift it."""
  contents = tf.read_file(f)
  raw = tf.image.decode_image(contents)
  converted = tf.image.convert_image_dtype(raw, tf.float32)
  padded = tf.pad(converted, [[pady, pady], [padx, padx], [0, 0]])
  image = shift_image(padded, xshift, yshift)
  image.set_shape([None, None, 3])  # RGB images have 3 channels.
  return image


def pose_from_flag(flag):
  if flag:
    values = [float(x) for x in flag.replace(',', ' ').split()]
    assert len(values) == 12
    return [values[0:4], values[4:8], values[8:12], [0.0, 0.0, 0.0, 1.0]]
  else:
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def get_inputs(padx, pady):
  """Get images, poses and intrinsics in required format."""
  inputs = {}
  image1 = load_image(FLAGS.image1, padx, pady, 0, 0)
  image2 = load_image(FLAGS.image2, padx, pady, -FLAGS.xshift, -FLAGS.yshift)

  shape1_before_crop = tf.shape(image1)
  shape2_before_crop = tf.shape(image2)
  image1 = crop_to_multiple(image1, 16)
  image2 = crop_to_multiple(image2, 16)
  shape1_after_crop = tf.shape(image1)
  shape2_after_crop = tf.shape(image2)

  with tf.control_dependencies([
      tf.Assert(
          tf.reduce_all(
              tf.logical_and(
                  tf.equal(shape1_before_crop, shape2_before_crop),
                  tf.equal(shape1_after_crop, shape2_after_crop))), [
                      'Shape mismatch:', shape1_before_crop, shape2_before_crop,
                      shape1_after_crop, shape2_after_crop
                  ])
  ]):
    # Add batch dimension (size 1).
    image1 = image1[tf.newaxis, ...]
    image2 = image2[tf.newaxis, ...]

  pose_one = pose_from_flag(FLAGS.pose1)
  pose_two = pose_from_flag(FLAGS.pose2)
  if not FLAGS.pose2:
    pose_two[0][3] = -FLAGS.xoffset
    pose_two[1][3] = -FLAGS.yoffset
    pose_two[2][3] = -FLAGS.zoffset

  pose_one = build_matrix(pose_one)[tf.newaxis, ...]
  pose_two = build_matrix(pose_two)[tf.newaxis, ...]

  # Use pre-crop and pre-padding sizing when converting fx, fy. This way the
  # field of view gets modified by the cropping correctly.
  original_width = shape1_before_crop[1] - 2 * padx
  original_height = shape1_before_crop[0] - 2 * pady
  eventual_width = shape1_after_crop[1]
  eventual_height = shape1_after_crop[0]
  fx = tf.multiply(tf.to_float(original_width), FLAGS.fx)
  fy = tf.multiply(tf.to_float(original_height), FLAGS.fy)

  # The MPI code may fail if the principal point is not in the center.  In
  # reality cropping might have shifted it by half a pixel, but we'll ignore
  # that here.
  cx = tf.multiply(tf.to_float(eventual_width), 0.5)
  cy = tf.multiply(tf.to_float(eventual_height), 0.5)
  intrinsics = build_matrix([[fx, 0.0, cx], [0.0, fy, cy],
                             [0.0, 0.0, 1.0]])[tf.newaxis, ...]
  inputs['ref_image'] = image1
  inputs['ref_pose'] = pose_one
  inputs['src_images'] = tf.concat([image1, image2], axis=-1)
  inputs['src_poses'] = tf.stack([pose_one, pose_two], axis=1)
  inputs['intrinsics'] = intrinsics
  return inputs, original_width, original_height


def main(_):
  # Set up the inputs.
  # How much shall we pad the input images? We'll pad enough so that
  # (a) when we render output images we won't lose stuff at the edges
  # due to cropping, and (b) we can find a multiple of 16 size without
  # cropping into the original images.
  max_multiple = 0
  if FLAGS.render:
    render_list = [float(x) for x in FLAGS.render_multiples.split(',')]
    max_multiple = max(abs(float(m)) for m in render_list)
  pady = int(max_multiple * abs(FLAGS.yshift) + 8)
  padx = int(max_multiple * abs(FLAGS.xshift) + 8)

  print 'Padding inputs: padx=%d, pady=%d (max_multiple=%d)' % (padx, pady,
                                                                max_multiple)
  inputs, original_width, original_height = get_inputs(padx, pady)

  # MPI code requires images of known size. So we run the input part of the
  # graph now to find the size, which we can then set on the inputs.
  with tf.Session() as sess:
    dimensions, original_width, original_height = sess.run(
        [tf.shape(inputs['ref_image']), original_width, original_height])
  batch = 1
  channels = 3
  assert dimensions[0] == batch
  mpi_height = dimensions[1]
  mpi_width = dimensions[2]
  assert dimensions[3] == channels

  print 'Original size: width=%d, height=%d' % (original_width, original_height)
  print '     MPI size: width=%d, height=%d' % (mpi_width, mpi_height)

  inputs['ref_image'].set_shape([batch, mpi_height, mpi_width, channels])
  inputs['src_images'].set_shape([batch, mpi_height, mpi_width, channels * 2])

  # Build the MPI.
  model = MPI()
  psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                                FLAGS.num_psv_planes)
  mpi_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                                FLAGS.num_mpi_planes)
  outputs = model.infer_mpi(
      inputs['src_images'], inputs['ref_image'], inputs['ref_pose'],
      inputs['src_poses'], inputs['intrinsics'], FLAGS.which_color_pred,
      FLAGS.num_mpi_planes, psv_planes, FLAGS.test_outputs)

  saver = tf.train.Saver([var for var in tf.model_variables()])
  ckpt_dir = os.path.join(FLAGS.model_root, FLAGS.model_name)
  ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
  sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
  config = tf.ConfigProto()

  config.gpu_options.allow_growth = True
  print 'Inferring MPI...'
  with sv.managed_session(config=config) as sess:
    saver.restore(sess, ckpt_file)
    ins, outs = sess.run([inputs, outputs])

  # Render output images separately so as not to run out of memory.
  tf.reset_default_graph()
  renders = {}
  if FLAGS.render:
    print 'Rendering new views...'
    for index, multiple in enumerate(render_list):
      m = float(multiple)
      print '    offset: %s' % multiple
      pose = build_matrix([[1.0, 0.0, 0.0, -m * FLAGS.xoffset],
                           [0.0, 1.0, 0.0, -m * FLAGS.yoffset],
                           [0.0, 0.0, 1.0, -m * FLAGS.zoffset],
                           [0.0, 0.0, 0.0, 1.0]])[tf.newaxis, ...]
      image = model.deprocess_image(
          model.mpi_render_view(
              tf.constant(outs['rgba_layers']), pose, mpi_planes,
              tf.constant(ins['intrinsics'])))[0]
      unshifted = shift_image(image, m * FLAGS.xshift, m * FLAGS.yshift)
      cropped = crop_to_size(unshifted, original_width, original_height)

      with tf.Session() as sess:
        renders[multiple] = (index, sess.run(cropped))

  output_dir = FLAGS.output_dir
  if not tf.gfile.IsDirectory(output_dir):
    tf.gfile.MakeDirs(output_dir)

  print 'Saving results to %s' % output_dir

  # Write results to disk.
  for name, (index, image) in renders.items():
    write_image(output_dir + '/render_%02d_%s.png' % (index, name), image)

  if 'intrinsics' in FLAGS.test_outputs:
    with open(output_dir + '/intrinsics.txt', 'w') as fh:
      write_intrinsics(fh, ins['intrinsics'][0])

  if 'src_images' in FLAGS.test_outputs:
    for i in range(2):
      write_image(output_dir + '/src_image_%d.png' % i,
                  ins['src_images'][0, :, :, i * 3:(i + 1) * 3] * 255.0)
      if 'poses' in FLAGS.test_outputs:
        write_pose(output_dir + '/src_pose_%d.txt' % i, ins['src_poses'][0, i])

  if 'fgbg' in FLAGS.test_outputs:
    write_image(output_dir + '/foreground_color.png', outs['fg_image'][0])
    write_image(output_dir + '/background_color.png', outs['bg_image'][0])

  if 'blend_weights' in FLAGS.test_outputs:
    for i in range(FLAGS.num_mpi_planes):
      weight_img = outs['blend_weights'][0, :, :, i] * 255.0
      write_image(output_dir + '/foreground_weight_plane_%.3d.png' % i,
                  weight_img)

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

  with open(output_dir + '/README', 'w') as fh:
    fh.write(
        'This directory was generated by mpi_from_images. Command-line:\n\n')
    fh.write('%s \\\n' % sys.argv[0])
    for arg in sys.argv[1:-1]:
      fh.write('  %s \\\n' % arg)
    fh.write('  %s\n' % sys.argv[-1])

  print 'Done.'


if __name__ == '__main__':
  tf.app.run()
