# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-08-10 20:45:28
"""
from __future__ import absolute_import, division, print_function

import keras
from keras import backend as K
from keras import optimizers
from keras.applications import ResNet50
from keras.layers import (Dense, Dropout, Embedding,
                          Flatten, Input, Lambda, Add)
from keras.models import Model

from utils import gen_data_and_classes, generator_from_csv, set_trainable

(img_height, img_width) = (224, 224)
batch_size = 32

train_csv = 'train.csv'
valid_csv = 'valid.csv'
train_lines, valid_lines, classes = gen_data_and_classes(train_csv, valid_csv)
class_mapping = {c: i for i, c in enumerate(classes)}

pre_model = ResNet50(weights='imagenet', include_top=True,
                     input_shape=(img_height, img_width, 3))

last_layer = pre_model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
out_activation = Dense(1024, activation='relu', name='dense')(x)
out_activation = Dropout(0.5)(out_activation)
out = Dense(len(class_mapping), activation='softmax', name='output_layer')(out_activation)

# Two methods can be used:
# 1. Use sparse_categorial_crossentropy
# 2. if using categorical_crossentropy:
#    should use keras.utils.np_utils.to_categorical in generator to convert labels.
model = Model(inputs=pre_model.input, outputs=out)
set_trainable(model, ['res4', 'res5', 'dense', 'output_layer'], bn=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',  # optimizers.SGD(lr=1e-5),
              metrics=['accuracy'])

ntrain, nvalid = len(train_lines), len(valid_lines)

print("""Training set: %d images.
         Validation set: %d images.
      """ % (ntrain, nvalid))

nbatches_train, _ = divmod(ntrain, batch_size)
nbatches_valid, _ = divmod(nvalid, 1)

filepath = "resnet50-{val_loss:.3f}-{val_acc:.3f}.h5"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-5),
]

train_generator = generator_from_csv(
    train_lines, class_mapping, batch_size=batch_size, target_size=(img_height, img_width))
validation_generator = generator_from_csv(
    valid_lines, class_mapping, batch_size=1, target_size=(img_height, img_width))
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=int(nbatches_train * 1.2),
    epochs=50,
    verbose=1,
    max_queue_size=50,
    validation_data=validation_generator,
    validation_steps=nbatches_valid,
    callbacks=callbacks,
    workers=4,
)

model.save_weights('model_final.h5')
