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
"""Functions for learning multiplane images (MPIs).
"""

from __future__ import division
import os
import time
import tensorflow as tf
import geometry.projector as pj
from third_party.vgg import build_vgg19
from nets import mpi_net


class MPI(object):
  """Class definition for MPI learning module.
  """

  def __init__(self):
    pass

  def infer_mpi(self,
                raw_src_images,
                raw_ref_image,
                ref_pose,
                src_poses,
                intrinsics,
                which_color_pred,
                num_mpi_planes,
                psv_planes,
                extra_outputs=''):
    """Construct the MPI inference graph.

    Args:
      raw_src_images: stack of source images [batch, height, width, 3*#source]
      raw_ref_image: reference image [batch, height, width, 3]
      ref_pose: reference frame pose (world to camera) [batch, 4, 4]
      src_poses: source frame poses (world to camera) [batch, #source, 4, 4]
      intrinsics: camera intrinsics [batch, 3, 3]
      which_color_pred: method for predicting the color at each MPI plane (see
        README)
      num_mpi_planes: number of MPI planes to predict
      psv_planes: list of depth of plane sweep volume (PSV) planes
      extra_outputs: extra variables to output in addition to RGBA layers
    Returns:
      outputs: a collection of output tensors.
    """
    batch_size, img_height, img_width, _ = raw_src_images.get_shape().as_list()
    with tf.name_scope('preprocessing'):
      src_images = self.preprocess_image(raw_src_images)
      ref_image = self.preprocess_image(raw_ref_image)

    with tf.name_scope('format_network_input'):
      # Note: we assume the first src image/pose is the reference.
      net_input = self.format_network_input(ref_image, src_images[:, :, :, 3:],
                                            ref_pose, src_poses[:, 1:],
                                            psv_planes, intrinsics)

    with tf.name_scope('layer_prediction'):
      if which_color_pred == 'bg':
        # Our default model. The network predicts: 1) weights for blending
        # the background and foreground (reference source image) color images
        # at each plane, 2) the alphas at each plane. 3) a background color
        # image.
        mpi_pred = mpi_net(net_input, 3 + num_mpi_planes * 2)
        # Rescale blend_weights to (0, 1)
        blend_weights = (mpi_pred[:, :, :, :num_mpi_planes] + 1.) / 2.
        # Rescale alphas to (0, 1)
        alphas = (
            mpi_pred[:, :, :, num_mpi_planes:num_mpi_planes * 2] + 1.) / 2.
        bg_rgb = mpi_pred[:, :, :, -3:]
        fg_rgb = ref_image
        # Assemble into an MPI (rgba_layers)
        for i in range(num_mpi_planes):
          curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
          w = tf.expand_dims(blend_weights[:, :, :, i], -1)
          curr_rgb = w * fg_rgb + (1 - w) * bg_rgb
          curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
          if i == 0:
            rgba_layers = curr_rgba
          else:
            rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
        rgba_layers = tf.reshape(
            rgba_layers, [batch_size, img_height, img_width, num_mpi_planes, 4])

      if which_color_pred == 'fgbg':
        # Instead of using the reference source as the foreground image,
        # the network predicts an extra foreground image for blending with the
        # background.
        mpi_pred = mpi_net(net_input, 6 + num_mpi_planes * 2)
        # Rescale blend_weights to (0, 1)
        blend_weights = (mpi_pred[:, :, :, :num_mpi_planes] + 1.) / 2.
        # Rescale alphas to (0, 1)
        alphas = (
            mpi_pred[:, :, :, num_mpi_planes:num_mpi_planes * 2] + 1.) / 2.
        bg_rgb = mpi_pred[:, :, :, -6:-3]
        fg_rgb = mpi_pred[:, :, :, -3:]
        # Assemble into MPI (rgba_layers)
        for i in range(num_mpi_planes):
          curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
          w = tf.expand_dims(blend_weights[:, :, :, i], -1)
          curr_rgb = w * fg_rgb + (1 - w) * bg_rgb
          curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
          if i == 0:
            rgba_layers = curr_rgba
          else:
            rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
        rgba_layers = tf.reshape(
            rgba_layers, [batch_size, img_height, img_width, num_mpi_planes, 4])

      if which_color_pred == 'all':
        # The network directly outputs the color image at each MPI plane.
        rgba_layers = mpi_net(net_input, num_mpi_planes * 4)
        rgba_layers = tf.reshape(
            rgba_layers, [batch_size, img_height, img_width, num_mpi_planes, 4])
        color_layers = rgba_layers[:, :, :, :, :3]
        alpha_layers = rgba_layers[:, :, :, :, 3:]
        # Rescale alphas to (0, 1)
        alpha_layers = (alpha_layers + 1.) / 2.
        rgba_layers = tf.concat([color_layers, alpha_layers], axis=4)

      if which_color_pred == 'alpha_only':
        # No color image (or blending weights) is predicted by the network.
        # The reference source image is used as the color image at each MPI
        # plane.
        alphas = mpi_net(net_input, num_mpi_planes)
        # Rescale alphas to (0, 1)
        alphas = (alphas + 1.) / 2.
        rgb = ref_image
        for i in range(num_mpi_planes):
          curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
          curr_rgba = tf.concat([rgb, curr_alpha], axis=3)
          if i == 0:
            rgba_layers = curr_rgba
          else:
            rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
        rgba_layers = tf.reshape(
            rgba_layers, [batch_size, img_height, img_width, num_mpi_planes, 4])

      if which_color_pred == 'single':
        # The network predicts a single color image shared by all MPI planes.
        alphas_and_rgb = mpi_net(net_input, num_mpi_planes + 3)
        alphas = alphas_and_rgb[:, :, :, :num_mpi_planes]
        # Rescale alphas to (0, 1)
        alphas = (alphas + 1.) / 2.
        rgb = alphas_and_rgb[:, :, :, -3:]
        for i in range(num_mpi_planes):
          curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
          curr_rgba = tf.concat([rgb, curr_alpha], axis=3)
          if i == 0:
            rgba_layers = curr_rgba
          else:
            rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
        rgba_layers = tf.reshape(
            rgba_layers, [batch_size, img_height, img_width, num_mpi_planes, 4])

    # Collect output tensors
    pred = {}
    pred['rgba_layers'] = rgba_layers
    if 'blend_weights' in extra_outputs:
      pred['blend_weights'] = blend_weights
    if 'psv' in extra_outputs:
      pred['psv'] = net_input[:, :, :, 3:]
    if 'fgbg' in extra_outputs:
      pred['fg_image'] = self.deprocess_image(fg_rgb)
      pred['bg_image'] = self.deprocess_image(bg_rgb)
    return pred

  def mpi_render_view(self, rgba_layers, tgt_pose, planes, intrinsics):
    """Render a target view from an MPI representation.

    Args:
      rgba_layers: input MPI [batch, height, width, #planes, 4]
      tgt_pose: target pose to render from [batch, 4, 4]
      planes: list of depth for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      rendered view [batch, height, width, 3]
    """
    batch_size, _, _ = tgt_pose.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_homography(rgba_layers, intrinsics,
                                                   tgt_pose, depths)
    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite(proj_images_list)
    return output_image

  def build_train_graph(self,
                        inputs,
                        min_depth,
                        max_depth,
                        num_psv_planes,
                        num_mpi_planes,
                        which_color_pred='bg',
                        which_loss='pixel',
                        learning_rate=0.0002,
                        beta1=0.9,
                        vgg_model_file=None):
    """Construct the training computation graph.

    Args:
      inputs: dictionary of tensors (see 'input_data' below) needed for training
      min_depth: minimum depth for the plane sweep volume (PSV) and MPI planes
      max_depth: maximum depth for the PSV and MPI planes
      num_psv_planes: number of PSV planes for network input
      num_mpi_planes: number of MPI planes to infer
      which_color_pred: how to predict the color at each MPI plane
      which_loss: which loss function to use (vgg or pixel)
      learning_rate: learning rate
      beta1: hyperparameter for ADAM
      vgg_model_file: path to VGG weights (required when VGG loss is used)
    Returns:
      A train_op to be used for training.
    """
    with tf.name_scope('setup'):
      psv_planes = self.inv_depths(min_depth, max_depth, num_psv_planes)
      mpi_planes = self.inv_depths(min_depth, max_depth, num_mpi_planes)

    with tf.name_scope('input_data'):
      raw_tgt_image = inputs['tgt_image']
      raw_ref_image = inputs['ref_image']
      raw_src_images = inputs['src_images']
      tgt_pose = inputs['tgt_pose']
      ref_pose = inputs['ref_pose']
      src_poses = inputs['src_poses']
      intrinsics = inputs['intrinsics']
      _, num_source, _, _ = src_poses.get_shape().as_list()

    with tf.name_scope('inference'):
      num_mpi_planes = len(mpi_planes)
      pred = self.infer_mpi(raw_src_images, raw_ref_image, ref_pose, src_poses,
                            intrinsics, which_color_pred, num_mpi_planes,
                            psv_planes)
      rgba_layers = pred['rgba_layers']

    with tf.name_scope('synthesis'):
      rel_pose = tf.matmul(tgt_pose, tf.matrix_inverse(ref_pose))
      output_image = self.mpi_render_view(rgba_layers, rel_pose, mpi_planes,
                                          intrinsics)

    with tf.name_scope('loss'):
      if which_loss == 'vgg':
        # Normalized VGG loss (from
        # https://github.com/CQFIO/PhotographicImageSynthesis)
        def compute_error(real, fake):
          return tf.reduce_mean(tf.abs(fake - real))

        vgg_real = build_vgg19(raw_tgt_image, vgg_model_file)
        rescaled_output_image = (output_image + 1.) / 2. * 255.0
        vgg_fake = build_vgg19(
            rescaled_output_image, vgg_model_file, reuse=True)
        p0 = compute_error(vgg_real['input'], vgg_fake['input'])
        p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
        p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
        p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
        p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
        p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5
        total_loss = p0 + p1 + p2 + p3 + p4 + p5

      if which_loss == 'pixel':
        tgt_image = self.preprocess_image(raw_tgt_image)
        total_loss = tf.reduce_mean(tf.nn.l2_loss(output_image - tgt_image))

    with tf.name_scope('train_op'):
      train_vars = [var for var in tf.trainable_variables()]
      optim = tf.train.AdamOptimizer(learning_rate, beta1)
      grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
      train_op = optim.apply_gradients(grads_and_vars)

    # Summaries
    tf.summary.scalar('total_loss', total_loss)
    # Source images
    for i in range(num_source):
      src_image = raw_src_images[:, :, :, i * 3:(i + 1) * 3]
      tf.summary.image('src_image_%d' % i, src_image)
    # Output image
    tf.summary.image('output_image', self.deprocess_image(output_image))
    # Target image
    tf.summary.image('tgt_image', raw_tgt_image)
    # Reference image
    tf.summary.image('ref_image', raw_ref_image)
    # Predicted color and alpha layers
    for i in range(0, num_mpi_planes, 8):
      rgb = rgba_layers[:, :, :, i, :3]
      alpha = rgba_layers[:, :, :, i, 3:]
      tf.summary.image('rgb_layer_%d' % i, self.deprocess_image(rgb))
      tf.summary.image('alpha_layer_%d' % i, alpha)
      tf.summary.image('rgba_layer_%d' % i, self.deprocess_image(rgb * alpha))

    return train_op

  def train(self, train_op, checkpoint_dir, continue_train, summary_freq,
            save_latest_freq, max_steps):
    """Runs the training procedure.

    Args:
      train_op: op for training the network
      checkpoint_dir: where to save the checkpoints and summaries
      continue_train: whether to restore training from previous checkpoint
      summary_freq: summary frequency
      save_latest_freq: Frequency of model saving (overwrites old one)
      max_steps: maximum training steps
    """
    parameter_count = tf.reduce_sum(
        [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    global_step = tf.Variable(0, name='global_step', trainable=False)
    incr_global_step = tf.assign(global_step, global_step + 1)
    saver = tf.train.Saver(
        [var for var in tf.model_variables()] + [global_step], max_to_keep=10)
    sv = tf.train.Supervisor(
        logdir=checkpoint_dir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
      tf.logging.info('Trainable variables: ')
      for var in tf.trainable_variables():
        tf.logging.info(var.name)
      tf.logging.info('parameter_count = %d' % sess.run(parameter_count))
      if continue_train:
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint is not None:
          tf.logging.info('Resume training from previous checkpoint')
          saver.restore(sess, checkpoint)
      for step in range(1, max_steps):
        start_time = time.time()
        fetches = {
            'train': train_op,
            'global_step': global_step,
            'incr_global_step': incr_global_step,
        }
        if step % summary_freq == 0:
          fetches['summary'] = sv.summary_op

        results = sess.run(fetches)
        gs = results['global_step']

        if step % summary_freq == 0:
          sv.summary_writer.add_summary(results['summary'], gs)
          tf.logging.info(
              '[Step %.8d] time: %4.4f/it' % (gs, time.time() - start_time))

        if step % save_latest_freq == 0:
          tf.logging.info(' [*] Saving checkpoint to %s...' % checkpoint_dir)
          saver.save(sess, os.path.join(checkpoint_dir, 'model.latest'))

  def format_network_input(self, ref_image, psv_src_images, ref_pose,
                           psv_src_poses, planes, intrinsics):
    """Format the network input (reference source image + PSV of the 2nd image).

    Args:
      ref_image: reference source image [batch, height, width, 3]
      psv_src_images: stack of source images (excluding the ref image)
                      [batch, height, width, 3*(num_source -1)]
      ref_pose: reference world-to-camera pose (where PSV is constructed)
                [batch, 4, 4]
      psv_src_poses: input poses (world to camera) [batch, num_source-1, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      net_input: [batch, height, width, (num_source-1)*#planes*3 + 3]
    """
    _, num_psv_source, _, _ = psv_src_poses.get_shape().as_list()
    net_input = []
    net_input.append(ref_image)
    for i in range(num_psv_source):
      curr_pose = tf.matmul(psv_src_poses[:, i], tf.matrix_inverse(ref_pose))
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = pj.plane_sweep(curr_image, planes, curr_pose, intrinsics)
      net_input.append(curr_psv)
    net_input = tf.concat(net_input, axis=3)
    return net_input

  def preprocess_image(self, image):
    """Preprocess the image for CNN input.

    Args:
      image: the input image in either float [0, 1] or uint8 [0, 255]
    Returns:
      A new image converted to float with range [-1, 1]
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image * 2 - 1

  def deprocess_image(self, image):
    """Undo the preprocessing.

    Args:
      image: the input image in float with range [-1, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    image = (image + 1.) / 2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

  def inv_depths(self, start_depth, end_depth, num_depths):
    """Sample reversed, sorted inverse depths between a near and far plane.

    Args:
      start_depth: The first depth (i.e. near plane distance).
      end_depth: The last depth (i.e. far plane distance).
      num_depths: The total number of depths to create. start_depth and
          end_depth are always included and other depths are sampled
          between them uniformly according to inverse depth.
    Returns:
      The depths sorted in descending order (so furthest first). This order is
      useful for back to front compositing.
    """
    inv_start_depth = 1.0 / start_depth
    inv_end_depth = 1.0 / end_depth
    depths = [start_depth, end_depth]
    for i in range(1, num_depths - 1):
      fraction = float(i) / float(num_depths - 1)
      inv_depth = inv_start_depth + (inv_end_depth - inv_start_depth) * fraction
      depths.append(1.0 / inv_depth)
    depths = sorted(depths)
    return depths[::-1]
