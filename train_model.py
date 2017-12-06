# @Author: Lee Yam Keng
# @Date: 2017-10-25 3:12:16
# @Last Modified by: Lee
# @Last Modified time: 2017-10-26 16:50:12

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random
import os

## Configuration and Hyperparameters

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

fc_size = 130

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# class info
num_classes = 2

# batch size
batch_size = 1

# validation split
validation_size = .1

# how long to wait after validation loss stops improving before terminating training
early_stopping = None    # use None if you don't want to implement early stopping

checkpoint_dir = 'model/'

# Load data
data = dataset.read_train_sets(batch_size = batch_size, validation_size = validation_size)

print ("Size of:")
print ("- Training-set:\t\t{}" . format(len(data.train.output)))
print ("- Validation-set:\t{}" . format(len(data.valid.output)))

D = len(data.train.input[1])
VD = len(data.valid.input[1])

for i in range(len(data.valid.output)):
	pass
	print ('i = ', i, 'item = ', data.valid.output[i])

print ('Input dimension', data.valid.input) # (5394, 2655)
print ('Input valid dimension', data.valid.output) # (5394,)

num_channels = 1

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape = [length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling = True):

	shape = [filter_size, filter_size, num_input_channels, num_filters]

	weights = new_weights(shape = shape)

	biases = new_biases(length = num_filters)

	layer = tf.nn.conv2d(input = input,
						 filter = weights,
						 strides = [1, 1, 1, 1],
						 padding = 'SAME')
	layer += biases

	if use_pooling:
		layer = tf.nn.max_pool(value = layer,
							   ksize = [1, 2, 2, 1],
							   strides = [1, 2, 2, 1],
							   padding = 'SAME')
	layer = tf.nn.relu(layer)

	return layer, weights

def flatten_layer(layer):

	layer_shape = layer.get_shape()

	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

def new_fc_layer(input,	num_inputs,	num_outputs, use_relu = True):

	weights = new_weights(shape = [num_inputs, num_outputs])
	biases = new_biases(length = num_outputs)

	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

# Get some random images and their labels from the train set.

# images, cls_true = data.train.images, data.train.cls

x = tf.placeholder(tf.float32, shape = [None, D], name = 'x')

x_image = tf.reshape(x, [-1, D, 1, 1])

y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')

y_true_cls = tf.argmax(y_true, dimension = 1)

# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input = x_image,
											num_input_channels = num_channels,
											filter_size = filter_size1,
											num_filters = num_filters1,
											use_pooling = True)

# Convolutional Layers 2 and 3
layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1,
											num_input_channels = num_filters1,
											filter_size = filter_size2,
											num_filters = num_filters2,
											use_pooling = True)

layer_conv3, weights_conv3 = new_conv_layer(input = layer_conv2,
											num_input_channels = num_filters2,
											filter_size = filter_size3,
											num_filters = num_filters3,
											use_pooling = True)

# Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv3)

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input = layer_flat,
						 num_inputs = num_features,
						 num_outputs = fc_size,
						 use_relu = True)

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input = layer_fc1,
						 num_inputs = fc_size,
						 num_outputs = num_classes,
						 use_relu = False)

# Predicted Class
y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2,
														labels = y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

# Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## TensorFlow Run

# Create TensorFlow session
session = tf.Session()

session.run(tf.global_variables_initializer())
train_batch_size = batch_size

saver = tf.train.Saver()

# if os.path.isfile('models/model/model.ckpt.index'):
# 	with tf.Session() as sess:
# 		# Restore variables from disk.
# 		saver.restore(session, "models/model/model.ckpt")
# 		print("Model restored.")

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
	acc = session.run(accuracy, feed_dict = feed_dict_train)
	val_acc = session.run(accuracy, feed_dict = feed_dict_validate)
	msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
	print (msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

def optimize(num_iterations):

	global total_iterations

	start_time = time.time()
	best_val_loss = float("inf")
	patience = 0

	for i in range(total_iterations,
				   total_iterations + num_iterations):
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		x_valid_batch, y_valid_batch = data.valid.next_batch(train_batch_size)

		x_batch = x_batch.reshape(train_batch_size, D)
		x_valid_batch = x_valid_batch.reshape(train_batch_size, D)

		feed_dict_train = {x: x_batch,
						   y_true: y_true_batch}

		feed_dict_validate = {x: x_valid_batch,
							  y_true: y_valid_batch}

		session.run(optimizer, feed_dict = feed_dict_train)

		if i % int(data.train.num_examples / batch_size) == 0:
			
			val_loss = session.run(cost, feed_dict = feed_dict_validate)
			epoch = int(i / int(data.train.num_examples / batch_size))
			print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

			if early_stopping:
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					patience = 0
				else:
					patience += 1
				if patience == early_stopping:
					break

	total_iterations += num_iterations

	end_time = time.time()
	time_dif = end_time - start_time

	print("Time elapsed: " + str(timedelta(seconds = int(round(time_dif)))))

def plot_example_errors(cls_pred, correct):

	incorrect = (correct == False)
	images = data.valid.images[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = data.valid.cls[incorrect]

	plot_images(images = images[0:9],
				cls_true = cls_true[0:9],
				cls_pred = cls_pred[0:9])

def plot_confusion_matrix(cls_pred):

	cls_true = data.valid.cls
	cm = confusion_matrix(y_true = cls_true,
						  y_pred = cls_pred)

	print (cm)
	plt.matshow(cm)

	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes))
	plt.yticks(tick_marks, range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')

	plt.show()

def print_validation_accuracy(show_example_errors = False, show_confusion_matrix = False):

	num_test = len(data.valid.input)
	cls_pred = np.zeros(shape = num_test, dtype = np.int)

	i = 0

	print ('data.valid.output', data.valid.output)

	while i < num_test:

		j = min(i + batch_size, num_test)

		feed_dict = {x: data.valid.input[i:j, :],
					 y_true: data.valid.output[i:j, :]}

		print ('===================================================')
		print ('data.valid.input[i:j, :]', data.valid.input[i:j, :])
		print ('data.valid.output[i:j, :]', data.valid.output[i:j, :])

		cls_pred[i:j] = session.run(y_pred_cls, feed_dict = feed_dict)
		print ('cls_pred', cls_pred[i:j])
		i = j

	# Save the variables to disk.
	save_path = saver.save(session, "models/model/model.ckpt")
	print("Model saved in file: %s" % save_path)

	print ('cls_pred', cls_pred)
	# cls_true = np.array(data.valid.cls)
	# cls_pred = np.array([classes[x] for x in cls_pred])

	# correct = (cls_true == cls_pred)

	# correct_sum = correct.sum()
	# acc = float(correct_sum) / num_test
	# msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
	# print (msg.format(acc, correct_sum, num_test))

	# if show_example_errors:
	# 	print ("Example errors:")
	# 	plot_example_errors(cls_pred = cls_pred, correct = correct)

	# if show_confusion_matrix:
	# 	print ("Confusion Matrix:")
	# 	plot_confusion_matrix(cls_pred = cls_pred)

def print_test_images(images_sam, ids):

	num_test = len(images_sam)
	cls_pred = np.zeros(shape = num_test, dtype = np.int)

	i = 0

	while i < num_test:

		j = min(i + batch_size, num_test)
		images_sam_split = images_sam[i:j, :].reshape(batch_size, img_size_flat)

		feed_dict = {x: images_sam_split}

		cls_pred[i:j] = session.run(y_pred_cls, feed_dict = feed_dict)

		i = j

	cls_pred = np.array([classes[x] for x in cls_pred])

iterations = len(data.train.output) * 20
optimize(num_iterations = iterations)
print_validation_accuracy()
# print_test_images(test_images, test_ids)