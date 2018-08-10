# ResNet-finetune-with-CAM-in-Keras

Finetune a resnet in Keras and visualize the CAM

1. Place images in a folder, named `data`, according to its classes.

2. Run `gen_dataset.py` to generate `train.csv`, `valid.csv` and `class_count.txt`.

   The generated file `class_count.txt` can be used to analyze the data.

3. Run `train.py` to train models using the generated datasets.

   There is a function `generator_from_csv` in `utils.py` to generate a generator to be used by `model.fit_generator()` from csv files.

   There is a function `gen_data_and_classes` in `utils.py` to fetch datasets to csv_lines and generate `classes` from csv files.

   There is a function `set_trainable` in `utils.py` to set which parts of the pretrained model to be trainable.

4. Run `test.py` to test the trained model using valid datasets.

5. Run `resnet_cam.py` to visualize the cam of pictures in the `test` folder using the trained model.

   The `resnet_cam` program was borrowed and modified from [ResNetCAM-keras](https://github.com/alexisbcook/ResNetCAM-keras).

## Tips
Run `./train.sh GPU` to train models using GPU and save the training logs.

For example:

> Run `./train.sh 0` to train models using GPU 0.