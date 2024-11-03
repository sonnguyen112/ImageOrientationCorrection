# RotNet

This repository contains the code necessary to train and test convolutional neural networks (CNNs) for predicting the rotation angle of an image to correct its orientation. There are scripts to train two models, one on [MNIST](http://yann.lecun.com/exdb/mnist/) and another one on the [Google Street View dataset](http://crcv.ucf.edu/data/GMCP_Geolocalization/). Since the data for this application is generated on-the-fly, you can also train using your own images in a similar way. A detailed explanation of the code and motivation for this project can be found in [my blog](https://d4nst.github.io/).

## Requirements
The code mainly relies on [Keras](https://keras.io/#installation) to train and test the CNN models, and [OpenCV](https://pypi.python.org/pypi/opencv-python) for image manipulation.

The python version should be 3.9.

You can install all the required packages using pip: `pip install -r requirements.txt`

The recommended way to use Keras is with the TensorFlow backend. If you want to use it with the Theano backend you will need to make some minor modifications to the code to make it work.

## Train

Download [here](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019) and put all images in a directory data/mit_dataset and run the following command:

`python train/train_mit_indoor.py` to train on the MIT Indoor Scene dataset.

If you only want to test the models, you can download pre-trained versions [here](https://drive.google.com/file/d/0B9eNEi5uvOI1SjQ5M2tQY3ZMM1U/view?usp=sharing&resourcekey=0-fxeNvoCZNlUrpQkzqZmDzw).

## Test
You can evaluate the models and display examples using the provided Jupyter notebooks. Simply run `jupyter notebook` from the root directory and navigate to `test/test_mit_indoor.ipynb`.

You can use the `correct_rotation.py` script to correct the orientation of your own images. You can run it as follows:

`python correct_rotation.py <path_to_hdf5_model> <path_to_input_image_or_directory>`

You can also specify the following command line arguments:
- `-o, --output` to specify the output image or directory.
- `-b, --batch_size` to specify the batch size used to run the model.
- `-c, --crop` to crop out the black borders after rotating the images.


You can aslo test with UI app by running the following command: `python app.py`
Note: You need to put models file in `models` folder to test the app.
