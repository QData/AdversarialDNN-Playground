# Utility script to find images from each class
# Use:
#  python json_gen.py images_to_generate.csv

import csv               # reading input files
import sys               # exit/commandline args
import json              # output file
from numpy import argmax # one-hot -> class label


# I could write my own reader here, but let's just use the tensorflow code
from tensorflow.examples.tutorials.mnist import input_data

mnist_filename='../webapp/models/mnist-model.meta'
data_filename = '../webapp/models/MNIST_data'
mnist = input_data.read_data_sets(data_filename, one_hot=True)

print('Loaded MNIST data...')

by_class = {}
for idx in range(100):
  curr_class = argmax(mnist.test._labels[idx])
  if curr_class in by_class:
    by_class[curr_class].append(idx)
  else:
    by_class[curr_class] = [idx]

for clss, lst in by_class.items():
  print('{}: {}'.format(clss, lst))


