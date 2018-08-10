#!/usr/bin/python
# -*- coding: utf-8 -*-
import ast
import glob
import sys

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image
from matplotlib.font_manager import _rebuild
from tqdm import tqdm

_rebuild()

mpl.rcParams['font.sans-serif'] = ['SimHei']


def pretrained_path_to_tensor(img_path):
  # loads RGB image as PIL.Image.Image type
  img = image.load_img(img_path, target_size=(224, 224))
  # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
  x = image.img_to_array(img)
  # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
  x = np.expand_dims(x, axis=0)
  # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
  return preprocess_input(x)


def get_ResNet():
  # define ResNet50 model
  ori_model = ResNet50(include_top=False, input_shape=(224, 224, 3))

  # build a classifier model to put on top of the convolutional model
  x = ori_model.get_layer('avg_pool').output
  x = Flatten(name='flatten')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dropout(0.5)(x)
  out = Dense(6, activation='softmax', name='output_layer')(x)
  model = Model(inputs=ori_model.input, outputs=out)

  model.load_weights('model_final.h5')

  # get AMP layer weights
  all_amp_layer_weights = model.layers[-1].get_weights()[0]
  # extract wanted output
  ResNet_model = Model(inputs=model.input,
                       outputs=(model.layers[-6].output, model.layers[-1].output))
  return ResNet_model, all_amp_layer_weights


def ResNet_CAM(img_path, model, all_amp_layer_weights):
  # get filtered images from convolutional output + model prediction vector
  last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path))
  # change dimensions of last convolutional outpu tto 7 x 7 x 2048
  last_conv_output = np.squeeze(last_conv_output)
  # get model's prediction (number between 0 and 999, inclusive)
  pred = np.argmax(pred_vec)
  # bilinear upsampling to resize each filtered image to size of original image
  mat_for_mult = scipy.ndimage.zoom(
      last_conv_output, (64, 64, 1), order=1)  # dim: 224 x 224 x 2048
  # get AMP layer weights
  amp_layer_weights = all_amp_layer_weights[:, pred]  # dim: (2048,)
  # get class activation map for object class that is predicted to be in the image
  final_output = np.dot(mat_for_mult.reshape((448*448, 2048)),
                        amp_layer_weights).reshape(448, 448)  # dim: 224 x 224
  # return class activation map
  return final_output


def plot_ResNet_CAM(img_path, fig, dpi, ax, model, all_amp_layer_weights):
  # load image, convert BGR --> RGB, resize image to 224 x 224,
  im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (448, 448))
  # plot image
  fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
  ax.imshow(im, alpha=0.5)
  # get class activation map
  CAM = ResNet_CAM(img_path, model, all_amp_layer_weights)
  # plot class activation map
  ax.imshow(CAM, cmap='jet', alpha=0.5)
  # load the dictionary that identifies each ImageNet category to an index in the prediction vector
  with open('classes.txt') as imagenet_classes_file:
    ast.literal_eval(imagenet_classes_file.read())
  # obtain the predicted ImageNet category
  # ax.set_title(unicode(imagenet_classes_dict[pred]))
  ax.set_title('title')


if __name__ == '__main__':
  ResNet_model, all_amp_layer_weights = get_ResNet()
  img_path = sys.argv[1]
  images = glob.glob('test/*.jpg')

  dpi = 200
  for im in tqdm(images):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)

    plot_ResNet_CAM(im, fig, dpi, ax, ResNet_model, all_amp_layer_weights)

    out_name = im.replace('test', 'test_out')
    fig.savefig(out_name, dpi=dpi)
    plt.close('all')
