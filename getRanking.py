# @Author: Lee Yam Keng
# @Date: 2017-10-25 12:23:16
# @Last Modified by: Lee
# @Last Modified time: 2017-10-25 13:41:23

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import argparse
import dataset
import random
import operator

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the image file")
args = vars(ap.parse_args())
test_path = args["file"] # for example: 'data/test/download0000c8e0f0d34c2381889b65bf707d9b-120.png'

## Configuration and Hyperparameters

#image dimensions (only squares for now)
img_size = 110
num_channels = 3
batch_size = 1
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
num_classes = 2

# Load data
product_list, products = dataset.import_test_data()

print ('product_list', product_list)
print ('product_list', product_list.shape)

# Create TensorFlow session
session = tf.Session()
saver = tf.train.import_meta_graph('models/model/model.ckpt.meta')

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(session, "models/model/model.ckpt")
  print("Model restored.")

graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name('y_pred:0')

# Let's feed the images to the input placeholders
x = graph.get_tensor_by_name('x:0')
y_true = graph.get_tensor_by_name('y_true:0')
y_test = np.zeros((1, 2))

dim = product_list.shape[1]
ranking_list = []
for i in range(len(product_list) - 1):
	pass
	product = product_list[i].reshape(1, dim)
	feed_dict = {x: product, y_true: y_test}
	cls_pred = session.run(y_pred, feed_dict = feed_dict)
	index, value = max(enumerate(cls_pred[0]), key=operator.itemgetter(1))
	# print ('ranking is ', index, ': accuracy is ', value)
	item = [int(products[i]), cls_pred[0][0]]
	ranking_list.append(item)

ranking_array = np.array(ranking_list)
print ('ranking_array', ranking_array[ranking_array[:, 1].argsort()])
ranks = ranking_array[ranking_array[:, 1].argsort()]
print ('ranks', ranks[:,0])

#save Rankings
np.savetxt('ranking_result.csv', ranks, delimiter=',')