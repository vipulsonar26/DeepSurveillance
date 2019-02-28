import numpy as np
import argparse
import imutils
import time
import cv2
import math

import matplotlib.pyplot as plt

import sys
caffe_root = '/home/bas/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import os
import cv2


import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/bas/caffe/models/BagLaptop_alexnet/deploy.prototxt'
PRETRAINED = '/home/bas/caffe/models/BagLaptop_alexnet/caffe_alexnet_train_iter_450000.caffemodel'

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('/home/bas/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(227, 227))
print "successfully loaded classifier"

# test on a image
IMAGE_FILE = '/home/bas/caffe/data/BagLaptop/images/imgBag_1412.jpg'
input_image = caffe.io.load_image(IMAGE_FILE)
# predict takes any number of images,
# and formats them for the Caffe net automatically
pred = net.predict([input_image])
print pred
