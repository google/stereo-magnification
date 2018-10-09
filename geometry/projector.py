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

"""A collection of projection utility functions.
"""
from __future__ import division
import tensorflow as tf
import homography


# Note that there is a subtle bug in how pixel coordinates are treated during
# projection. The projection code assumes pixels are centered at integer
# coordinates. However, this implies that we need to treat the domain of images
# as [-0.5, W-0.5] x [-0.5, H-0.5], whereas we actually use [0, H-1] x [0,
# W-1]. The outcome is that the principal point is shifted by a half-pixel from
# where it should be. We do not believe this issue makes a significant
# difference to the results, however.

def projective_forward_homography(src_images, intrinsics, pose, depths):
  """Use homography for forward warping.

  Args:
    src_images: [layers, batch, height, width, channels]
    intrinsics: [batch, 3, 3]
    pose: [batch, 4, 4]
    depths: [layers, batch]
  Returns:
    proj_src_images: [layers, batch, height, width, channels]
  """
  n_layers, n_batch, height, width, _ = src_images.get_shape().as_list()
  # Format for planar_transform code:
  # rot: relative rotation, [..., 3, 3] matrices
  # t: [B, 3, 1], translations from source to target camera (R*p_s + t = p_t)
  # n_hat: [L, B, 1, 3], plane normal w.r.t source camera frame [0,0,1]
  #        in our case
  # a: [L, B, 1, 1], plane equation displacement (n_hat * p_src + a = 0)
  rot = pose[:, :3, :3]
  t = pose[:, :3, 3:]
  n_hat = tf.constant([0., 0., 1.], shape=[1, 1, 1, 3])
  n_hat = tf.tile(n_hat, [n_layers, n_batch, 1, 1])
  a = -tf.reshape(depths, [n_layers, n_batch, 1, 1])
  k_s = intrinsics
  k_t = intrinsics
  pixel_coords_trg = tf.transpose(
      meshgrid_abs(n_batch, height, width), [0, 2, 3, 1])
  proj_src_images = homography.planar_transform(
      src_images, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
  return proj_src_images


def meshgrid_abs(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid in the absolute coordinates.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  xs = tf.linspace(0.0, tf.cast(width-1, tf.float32), width)
  ys = tf.linspace(0.0, tf.cast(height-1, tf.float32), height)
  xs, ys = tf.meshgrid(xs, ys)

  if is_homogeneous:
    ones = tf.ones_like(xs)
    coords = tf.stack([xs, ys, ones], axis=0)
  else:
    coords = tf.stack([xs, ys], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords


def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  xy_u = unnormalized_pixel_coords[:, 0:2, :]
  z_u = unnormalized_pixel_coords[:, 2:3, :]
  pixel_coords = xy_u / (z_u + 1e-10)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def projective_inverse_warp(
    img, depth, pose, intrinsics, ret_flows=False):
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
    ret_flows: whether to return the displacements/flows as well
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch, height, width, _ = img.get_shape().as_list()
  # Construct pixel grid coordinates.
  pixel_coords = meshgrid_abs(batch, height, width)

  # Convert pixel coordinates to the camera frame.
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)

  # Construct a 4x4 intrinsic matrix.
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)

  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)

  output_img = tf.contrib.resampler.resampler(img, src_pixel_coords)
  if ret_flows:
    return output_img, src_pixel_coords - cam_coords
  else:
    return output_img


def over_composite(rgbas):
  """Combines a list of RGBA images using the over operation.

  Combines RGBA images from back to front with the over operation.
  The alpha image of the first image is ignored and assumed to be 1.0.

  Args:
    rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
  Returns:
    Composited RGB image.
  """
  for i in range(len(rgbas)):
    rgb = rgbas[i][:, :, :, 0:3]
    alpha = rgbas[i][:, :, :, 3:]
    if i == 0:
      output = rgb
    else:
      rgb_by_alpha = rgb * alpha
      output = rgb_by_alpha + output * (1.0 - alpha)
  return output


def plane_sweep(img, depth_planes, pose, intrinsics):
  """Construct a plane sweep volume.

  Args:
    img: source image [batch, height, width, #channels]
    depth_planes: a list of depth values for each plane
    pose: target to source camera transformation [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    A plane sweep volume [batch, height, width, #planes*#channels]
  """
  batch, height, width, _ = img.get_shape().as_list()
  plane_sweep_volume = []

  for depth in depth_planes:
    curr_depth = tf.constant(
        depth, dtype=tf.float32, shape=[batch, height, width])
    warped_img = projective_inverse_warp(img, curr_depth, pose, intrinsics)
    plane_sweep_volume.append(warped_img)
  plane_sweep_volume = tf.concat(plane_sweep_volume, axis=3)
  return plane_sweep_volume
