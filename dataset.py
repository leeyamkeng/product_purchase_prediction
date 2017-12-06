# @Author: Lee Yam Keng
# @Date: 2017-10-25 3:12:16
# @Last Modified by: Lee
# @Last Modified time: 2017-10-25 20:41:23

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import numpy as np
import pandas
from sklearn.utils import shuffle
from pre_processing import processing

# standardize name of teams
def mapping(id, product):
  
  global normalized_values
  # training dataset generation 
  # normalized_values = np.eye(len(product), dtype=int)
  index = product.index(id)
  # row = np.zeros(len(product))
  # row[index] = 1
  # return row
  return normalized_values[index]

def getdiff(ids):

  list = []
  for item in ids:
    if item in list:
      pass
    else:
      list.append(item)

  return list

def normalization(data, product):

  dataset = []
  
  for i in range(0, len(data)):
    encoded_product = mapping(data[i, 0], product)
    item = np.concatenate((data[i], encoded_product))
    dataset.append(item)
  return np.array(dataset)

def import_data():

  # dataframe = pandas.read_csv('dataset/integrated_table.csv', engine = "python").values
  dataframe = processing()

  # dataframe = dataframe.values
  new_list = []
  old_mcvisid = ''
  old_time = ''
  dataframe = dataframe.values
  for i in range(len(dataframe)):
    if dataframe[i][0] == old_mcvisid and dataframe[i][1] == old_time:
      pass
    else:
      if i == 0:
        continue
      else:
        if pandas.isinstanceull(dataframe[i - 1][2]):
          continue
        else:
          new_list.append(dataframe[i - 1])
      old_mcvisid = dataframe[i][0]
      old_time = dataframe[i][1]

  print ("new_list", np.array(new_list)[2])
  # exit()

  dataset = np.array(new_list)

  # delete mcvisid and timestamp
  data_raw = np.delete(dataset, [0, 1], 1)

  output = data_raw[:, 1]
  labels = []
  for i in range(len(output)):
    pass
    label = np.zeros(2)
    label[output[i]] = 1.0
    labels.append(label)
  
  input = np.delete(data_raw, [1], 1)
  print ("input", input[2])
  print ("input.shape", input.shape)
  
  input = np.delete(input, 0, 1)
  print ("input", input[2])
  print ("output", output[2])
  exit()

  return input, np.array(labels)

def import_test_data():

  dataframe = pandas.read_csv('dataset/integrated_table.csv', engine = "python").values

  new_list = []
  old_mcvisid = ''
  old_time = ''
  print ('dataset', dataframe.shape)
  for i in range(int(len(dataframe) / 2)):
    if dataframe[i][1] == old_mcvisid and dataframe[i][2] == old_time:
      pass
    else:
      if i == 0:
        continue
      else:
        if pandas.isnull(dataframe[i - 1][4]):
          continue
        else:
          new_list.append(dataframe[i - 1])
      old_mcvisid = dataframe[i][1]
      old_time = dataframe[i][2]

  print ("new_list.shape", np.array(new_list).shape)
  dataset = np.array(new_list)

  data_raw = np.delete(dataset, [0, 1, 2, 3, 5, 6, 8, dataset.shape[1] - 1, dataset.shape[1] - 2, dataset.shape[1] - 3, dataset.shape[1] - 4], 1)
  products = getdiff(data_raw[:, 0])

  products_list = []
  for i in range(len(products) - 1):
    pass
    products_list.append(data_raw[np.where(data_raw[:, 0] == products[i])][0])

  print ('products_list', np.array(products_list).shape)

  products_list = np.delete(np.array(products_list), [1], 1)

  products_list = normalization(products_list, products)
  products_list = np.delete(products_list, 0, 1)

  return products_list, products

class DataSet(object):

  def __init__(self, input, output):

    self._num_examples = input.shape[0]

    self._input = input
    self._output = output
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def input(self):
    return self._input

  @property
  def output(self):
    return self._output

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._input[start:end], self._output[start:end]


def read_train_sets(batch_size, validation_size=0):

  class DataSets(object):
    pass
  data_sets = DataSets()

  input, output = import_data()
  print ('imported ....')
  input, output = shuffle(input, output)
  print ('shuffled ....')
  if isinstance(validation_size, float):
    validation_size = int(int(validation_size * input.shape[0]) / batch_size) * batch_size

  validation_input = input[:validation_size]
  validation_output = output[:validation_size]

  train_input = input[validation_size:]
  train_output = output[validation_size:]

  data_sets.train = DataSet(train_input, train_output)
  data_sets.valid = DataSet(validation_input, validation_output)

  return data_sets


def read_test_set(test_path, image_size):
  input, ids  = load_test(test_path, image_size)
  return input, ids
