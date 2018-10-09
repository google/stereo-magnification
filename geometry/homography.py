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

"""TensorFlow utils for image transformations via homographies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import geometry.sampling


def divide_safe(num, den, name=None):
  eps = 1e-8
  den += eps * tf.cast(tf.equal(den, 0), 'float32')
  return tf.divide(num, den, name=name)


def inv_homography(k_s, k_t, rot, t, n_hat, a):
  """Computes inverse homography matrix between two cameras via a plane.

  Args:
      k_s: intrinsics for source cameras, [..., 3, 3] matrices
      k_t: intrinsics for target cameras, [..., 3, 3] matrices
      rot: relative rotations between source and target, [..., 3, 3] matrices
      t: [..., 3, 1], translations from source to target camera. Mapping a 3D
        point p from source to target is accomplished via rot * p + t.
      n_hat: [..., 1, 3], plane normal w.r.t source camera frame
      a: [..., 1, 1], plane equation displacement
  Returns:
      homography: [..., 3, 3] inverse homography matrices (homographies mapping
        pixel coordinates from target to source).
  """
  with tf.name_scope('inv_homography'):
    rot_t = tf.matrix_transpose(rot)
    k_t_inv = tf.matrix_inverse(k_t, name='k_t_inv')

    denom = a - tf.matmul(tf.matmul(n_hat, rot_t), t)
    numerator = tf.matmul(tf.matmul(tf.matmul(rot_t, t), n_hat), rot_t)
    inv_hom = tf.matmul(
        tf.matmul(k_s, rot_t + divide_safe(numerator, denom)),
        k_t_inv, name='inv_hom')
    return inv_hom


def transform_points(points, homography):
  """Transforms input points according to homography.

  Args:
      points: [..., H, W, 3]; pixel (u,v,1) coordinates.
      homography: [..., 3, 3]; desired matrix transformation
  Returns:
      output_points: [..., H, W, 3]; transformed (u,v,w) coordinates.
  """
  with tf.name_scope('transform_points'):
    # Because the points have two additional dimensions as they vary across the
    # width and height of an image, we need to reshape to multiply by the
    # per-image homographies.
    points_orig_shape = points.get_shape().as_list()
    points_reshaped_shape = homography.get_shape().as_list()
    points_reshaped_shape[-2] = -1

    points_reshaped = tf.reshape(points, points_reshaped_shape)
    transformed_points = tf.matmul(points_reshaped, homography, transpose_b=True)
    transformed_points = tf.reshape(transformed_points, points_orig_shape)
    return transformed_points


def normalize_homogeneous(points):
  """Converts homogeneous coordinates to regular coordinates.

  Args:
      points: [..., n_dims_coords+1]; points in homogeneous coordinates.
  Returns:
      points_uv_norm: [..., n_dims_coords];
          points in standard coordinates after dividing by the last entry.
  """
  with tf.name_scope('normalize_homogeneous'):
    uv = points[..., :-1]
    w = tf.expand_dims(points[..., -1], -1)
    return divide_safe(uv, w)


def transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
  """Transforms input imgs via homographies for corresponding planes.

  Args:
    imgs: are [..., H_s, W_s, C]
    pixel_coords_trg: [..., H_t, W_t, 3]; pixel (u,v,1) coordinates.
    k_s: intrinsics for source cameras, [..., 3, 3] matrices
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotation, [..., 3, 3] matrices
    t: [..., 3, 1], translations from source to target camera
    n_hat: [..., 1, 3], plane normal w.r.t source camera frame
    a: [..., 1, 1], plane equation displacement
  Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
      Coordinates outside the image are sampled as 0.
  """
  with tf.name_scope('transform_plane_imgs'):
    hom_t2s_planes = inv_homography(k_s, k_t, rot, t, n_hat, a)
    pixel_coords_t2s = transform_points(pixel_coords_trg, hom_t2s_planes)
    pixel_coords_t2s = normalize_homogeneous(pixel_coords_t2s)
    imgs_s2t = geometry.sampling.bilinear_wrapper(imgs, pixel_coords_t2s)

    return imgs_s2t


def planar_transform(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
  """Transforms imgs, masks and computes dmaps according to planar transform.

  Args:
    imgs: are [L, B, H, W, C], typically RGB images per layer
    pixel_coords_trg: tensors with shape [B, H_t, W_t, 3];
        pixel (u,v,1) coordinates of target image pixels. (typically meshgrid)
    k_s: intrinsics for source cameras, [B, 3, 3] matrices
    k_t: intrinsics for target cameras, [B, 3, 3] matrices
    rot: relative rotation, [B, 3, 3] matrices
    t: [B, 3, 1] matrices, translations from source to target camera
       (R*p_src + t = p_tgt)
    n_hat: [L, B, 1, 3] matrices, plane normal w.r.t source camera frame
      (typically [0 0 1])
    a: [L, B, 1, 1] matrices, plane equation displacement
      (n_hat * p_src + a = 0)
  Returns:
    imgs_transformed: [L, ..., C] images in trg frame
  Assumes the first dimension corresponds to layers.
  """
  with tf.name_scope('planar_transform'):
    n_layers = imgs.get_shape().as_list()[0]
    rot_rep_dims = [n_layers]
    rot_rep_dims += [1 for _ in range(len(k_s.get_shape()))]

    cds_rep_dims = [n_layers]
    cds_rep_dims += [1 for _ in range(len(pixel_coords_trg.get_shape()))]

    k_s = tf.tile(tf.expand_dims(k_s, axis=0), rot_rep_dims)
    k_t = tf.tile(tf.expand_dims(k_t, axis=0), rot_rep_dims)
    t = tf.tile(tf.expand_dims(t, axis=0), rot_rep_dims)
    rot = tf.tile(tf.expand_dims(rot, axis=0), rot_rep_dims)
    pixel_coords_trg = tf.tile(tf.expand_dims(
        pixel_coords_trg, axis=0), cds_rep_dims)

    imgs_trg = transform_plane_imgs(
        imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return imgs_trg
