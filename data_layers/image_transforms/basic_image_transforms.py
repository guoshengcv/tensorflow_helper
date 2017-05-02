from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib
import numpy as np
import six


__all__ = [
  'resize_image',
  'pad_to_bounding_box',
  'resize_image_with_crop_or_pad',
  'random_crop',
  'central_crop',
  'sample_distorted_bounding_box',
  'flip_left_right',
  'flip_up_down',
  'adjust_brightness',
  'random_brightness',
  'adjust_contrast',
  'random_contrast',
  'random_flip_left_right',
  'random_flip_up_down',
  'adjust_hue',
  'random_hue',
  'adjust_saturation',
  'random_saturation',
  'rot90',
  'rotate',
  'random_rotate',
  'adjust_gamma',
  'grayscale_to_rgb',
  'rgb_to_grayscale',
  'rgb_to_hsv',
  'hsv_to_rgb',
  'per_image_standardization'
]


def resize_image(size, method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  def fun(image, label):
    return (
      tf.image.resize_images(
        images = image,
        size = size,
        method = method,
        align_corners = align_corners
      ),
      label
    )
  return fun

def pad_to_bounding_box(offset_height, offset_width, target_height, target_width):
  def fun(image, label):
    return (
      tf.image.pad_to_bounding_box(
        image = image,
        offset_height = offset_height,
        offset_width = offset_width,
        target_height = target_height,
        target_width = target_width
      ),
      label
    )
  return fun

def resize_image_with_crop_or_pad(target_height, target_width):
  def fun(image, label):
    return (
      tf.image.resize_image_with_crop_or_pad(
        image = image,
        target_height = target_height,
        target_width = target_width
      ),
      label
    )
  return fun

def random_crop(size, seed=None):
  def fun(image, label):
    return (
      tf.random_crop(
        value = image,
        size = size,
        seed = seed
      ),
      label
    )
  return fun

def central_crop(central_fraction):
  def fun(image, label):
    return (
      tf.image.central_crop(
        image = image,
        central_fraction = central_fraction
      ),
      label
    )
  return fun

def sample_distorted_bounding_box(
    aspect_ratio_range = [0.75, 1.33],
    area_range = [0.05, 1.0],
    max_attempts = 100):
  def fun(image, label):
    begin, size, bboxes = tf.image.sample_distorted_bounding_box(
      image_size = tf.shape(image),
      bounding_boxes = [[[0.,0.,1.,1.]]],
      aspect_ratio_range = aspect_ratio_range,
      area_range = area_range,
      max_attempts = max_attempts,
      use_image_if_no_bounding_boxes = True
    )
    tf.summary.image(
      name = 'images_with_distorted_bounding_box',
      tensor = tf.image.draw_bounding_boxes(
        images = tf.expand_dims(image, 0),
        boxes = bboxes
      )
    )
    return (
      tf.slice(
        image,
        begin = begin,
        size = size
      ),
      label
    )
  return fun

def rot90(k):
  def fun(image, label):
    return (
      tf.image.rot90(
        image = image,
        k = k
      ),
      label
    )
  return fun

def rotate(angle):
  def fun(image, label):
    return (
      tf.contrib.image.rotate(
        images = image,
        angles = angle
      ),
      label
    )
  return fun

def random_rotate(max_angle=180, seed=None):
  def fun(image, label):
    angle = tf.random_uniform(
      shape = [],
      minval = -max_angle,
      maxval = max_angle,
      seed = seed
    )
    return (
      tf.contrib.image.rotate(
        images = image,
        angles = angle
      ),
      label
    )
  return fun

def flip_left_right():
  def fun(image, label):
    return (
      tf.image.flip_left_right(
        image = image
      ),
      label
    )
  return fun

def random_flip_left_right(seed=None):
  def fun(image, label):
    return (
      tf.image.random_flip_left_right(
        image = image,
        seed = seed
      ),
      label
    )
  return fun

def flip_up_down():
  def fun(image, label):
    return (
      tf.image.flip_up_down(
        image = image
      ),
      label
    )
  return fun

def random_flip_up_down(seed=None):
  def fun(image, label):
    return (
      tf.image.random_flip_up_down(
        image = image,
        seed = seed
      ),
      label
    )
  return fun

def adjust_brightness(delta):
  def fun(image, label):
    return (
      tf.image.adjust_brightness(
        image = image,
        delta = delta
      ),
      label
    )
  return fun

def random_brightness(max_delta, seed=None):
  def fun(image, label):
    return (
      tf.image.random_brightness(
        image = image,
        max_delta = max_delta,
        seed = seed
      ),
      label
    )
  return fun

def adjust_contrast(contrast_factor):
  def fun(image, label):
    return (
      tf.image.adjust_contrast(
        images = image,
        contrast_factor = contrast_factor
      ),
      label
    )
  return fun

def random_contrast(lower, upper, seed=None):
  def fun(image, label):
    return (
      tf.image.random_contrast(
        image = image,
        lower = lower,
        upper = upper,
        seed = seed
      ),
      label
    )
  return fun

def adjust_hue(delta):
  def fun(image, label):
    return (
      tf.image.adjust_hue(
        image = image,
        delta = delta
      ),
      label
    )
  return fun

def random_hue(max_delta, seed=None):
  def fun(image, label):
    return (
      tf.image.random_hue(
        image = image,
        max_delta = max_delta,
        seed = seed
      ),
      label
    )
  return fun

def adjust_saturation(saturation_factor):
  def fun(image, label):
    return (
      tf.image.adjust_saturation(
        saturation_factor = saturation_factor
      ),
      label
    )
  return fun

def random_saturation(lower, upper, seed=None):
  def fun(image, label):
    return (
      tf.image.random_saturation(
        image = image,
        lower = lower,
        upper = upper,
        seed = seed
      ),
      label
    )
  return fun

def adjust_gamma(gamma=1, gain=1):
  def fun(image, label):
    return (
      tf.image.adjust_gamma(
        image = image,
        gamma = gamma,
        gain = gain
      ),
      label
    )
  return fun

def per_image_standardization():
  def fun(image, label):
    return (
      tf.image.per_image_standardization(
        image = image
      ),
      label
    )
  return fun

def grayscale_to_rgb():
  def fun(image, label):
    return (
      tf.image.grayscale_to_rgb(
        images = image
      ),
      label
    )
  return fun

def rgb_to_grayscale():
  def fun(image, label):
    return (
      tf.image.rgb_to_grayscale(
        images = image
      ),
      label
    )
  return fun

def rgb_to_hsv():
  def fun(image, label):
    return (
      tf.image.rgb_to_hsv(
        images = image
      ),
      label
    )
  return fun

def hsv_to_rgb():
  def fun(image, label):
    return (
      tf.image.hsv_to_rgb(
        images = image
      ),
      label
    )
  return fun
