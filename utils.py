#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import threading

import keras
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


def gen_data_and_classes(train_csv, valid_csv):
  with open(train_csv, 'r') as f:
    train_lines = [line.strip().split(',') for line in f.readlines()]
  with open(valid_csv, 'r') as f:
    valid_lines = [line.strip().split(',') for line in f.readlines()]
  lines = train_lines + valid_lines
  classes = []
  for line in lines:
    cls = line[1]
    if cls not in classes:
      classes.append(cls)
  with open('classes.txt', 'w') as f:
    for cls in classes:
      f.write('{}\n'.format(cls))
  return train_lines, valid_lines, classes


def set_trainable(model, trainable_layers, bn=False):
  def contain(name, layers):
    for layer in layers:
      if layer in name:
        return True
    return False

  for layer in model.layers:
    if contain(layer.name, trainable_layers):
      layer.trainable = True
    else:
      layer.trainable = False
    if not bn and layer.name.startswith('bn'):
      layer.trainable = False


class threadsafe_iter(object):
  """Takes an iterator/generator and makes it thread-safe by
  serializing call to the `next` method of given iterator/generator.
  """

  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def next(self):
    with self.lock:
      return self.it.next()


def threadsafe_generator(f):
  """A decorator that takes a generator function and makes it thread-safe.
  """
  def g(*a, **kw):
    return threadsafe_iter(f(*a, **kw))
  return g


@threadsafe_generator
def generator_from_csv(lines, class_mapping, batch_size, target_size, one_hot=True):

  nbatches, _ = divmod(len(lines), batch_size)

  count = 1
  epoch = 0
  cls_no = len(class_mapping)

  while True:
    np.random.shuffle(lines)
    epoch += 1
    i, j = 0, batch_size

    for _ in range(nbatches):
      sub = lines[i:j]
      try:
        X = np.array([img_to_array(load_img(f[0], target_size=target_size))
                      for f in sub])
        X = preprocess_input(X)

        if one_hot:
          Y = keras.utils.np_utils.to_categorical(
              np.array([class_mapping[s[1]] for s in sub]), cls_no)
        else:
          Y = np.array([class_mapping[s[1]] for s in sub])

        yield X, Y

      except IOError:
        count -= 1

      i = j
      j += batch_size
      count += 1
