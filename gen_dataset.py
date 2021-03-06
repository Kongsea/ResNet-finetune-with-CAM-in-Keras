# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-08-10 19:31:47
"""
from __future__ import absolute_import, division, print_function

import csv
import os

import numpy as np

np.random.seed(0)
ratio = 0.9


def gen_lines(path, d):
  path = os.path.join(path, d)
  lines = []
  for f in os.listdir(path):
    if f.endswith('.jpg'):
      lines.append([os.path.join(path, f), d])
  return lines


path = 'data'
dirs = sorted(os.listdir(path))

all_lines = []
for d in dirs:
  all_lines.append(gen_lines(path, d))

class_count = {}
len_sum = 0
for lines in all_lines:
  print('{} -> {}'.format(lines[0][1], len(lines)))
  class_count[lines[0][1]] = len(lines)
  len_sum += len(lines)
len_avg = int(len_sum/len(all_lines))

cc = sorted(class_count, key=class_count.get, reverse=True)
with open('class_count.txt', 'w') as f:
  for c in cc:
    f.write('{} {}\n'.format(c, class_count[c]))

train_lines = []
valid_lines = []

for lines in all_lines:
  np.random.shuffle(lines)
  cut = int(len(lines)*ratio)
  train_lines += lines[:cut]
  valid_lines += lines[cut:]

np.random.shuffle(train_lines)
np.random.shuffle(valid_lines)

with open('train.csv', 'wb') as f:
  cw = csv.writer(f)
  for line in train_lines:
    cw.writerow(line)

with open('valid.csv', 'wb') as f:
  cw = csv.writer(f)
  for line in valid_lines:
    cw.writerow(line)
