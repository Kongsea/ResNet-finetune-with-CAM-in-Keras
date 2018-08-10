# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-08-10 20:45:08
"""
from __future__ import absolute_import, division, print_function

import shutil

import cv2
import keras
import keras.preprocessing.image
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import generator_from_csv


def get_session():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())

with open('valid.csv', 'r') as f:
  valid_lines = [line.strip().split(',') for line in f.readlines()]

model_path = 'model_final.h5'
model = keras.models.load_model(model_path)
print('Using {}'.format(model_path))

with open('classes.txt') as f:
  classes = [c.strip() for c in f.readlines()]
cls = {c: i for i, c in enumerate(classes)}
cls_inv = {v: k for k, v in cls.items()}

results = {}
error_cls = {}
not_sure = {}
for line in valid_lines:
  path = line[0]
  label = cls[line[1]]
  image = cv2.imread(path)
  image = cv2.resize(image, (320, 320))
  image = 2 * (image / 255. - 0.5)
  image = np.expand_dims(image, axis=0)
  pred = model.predict_on_batch(image)
  results.setdefault(label, []).append(np.argmax(pred))
  if np.argmax(pred) != label:
    error_cls.setdefault(label, []).append(
        (path, cls_inv[np.argmax(pred)], pred[0, np.argmax(pred)]))
  if pred[0, np.argmax(pred)] <= 0.8:
    not_sure.setdefault(label, []).append(
        (path, cls_inv[np.argmax(pred)], pred[0, np.argmax(pred)]))


result = {}
all_correct = 0
all_counts = 0
for k, v in results.items():
  _acorrect = np.sum(np.array(v) == k)
  _acounts = len(v)
  result[cls_inv[k]] = _acorrect / _acounts
  all_correct += _acorrect
  all_counts += _acounts
result['all'] = all_correct / all_counts

for k, v in result.items():
  print('{} -> {:.3f}'.format(k, v))

for k, v in error_cls.items():
  print('Valid Error in {}'.format(cls_inv[k]))
  for f, p, pr in v:
    print('{}: {} -> {}'.format(f, cls_inv[k], p))
    shutil.copy(
        f, 'data/misclassified_keras/val_{}_{}_{}_{:.3f}.jpg'.format(f.rpartition('/')[-1][:-4], cls_inv[k], p, pr))

for k, v in not_sure.items():
  print('Not Sure in {}'.format(cls_inv[k]))
  for f, p, pr in v:
    print('{}: {} -> {} with {:.3f}'.format(f, cls_inv[k], p, pr))
    shutil.copy(
        f, 'data/not_sure/val_{}_{}_{}_{:.3f}.jpg'.format(f.rpartition('/')[-1][:-4], cls_inv[k], p, pr))
