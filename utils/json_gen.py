# Utility script to generate the seeds.json file
# Use:
#  python json_gen.py images_to_generate.csv

import os
import csv               # reading input files
import sys               # exit/commandline args
import json              # output file
import tensorflow as tf  # for classification
from numpy import argmax # one-hot -> class label
from scipy.misc import imsave # generating images 

help_text = (  'This script generates the "seeds.json" file for use in the ' 
             + 'webapp, \nallowing a developer to select which images they\'d '
             + 'like to include in the\noptions for the user.')
print(help_text)

from_file = len(sys.argv) > 1
if from_file:
  print('Input file detected...')
  try:
    with open(sys.argv[1]) as f:
      reader = csv.DictReader(f)
      indices = []
      display_names = []
      for row in reader:
        indices.append(int(row['index']))
        display_names.append(row['description'])
  except Exception as e:
    print('Something went wrong reading the file:')
    print(e)
    sys.exit(-1)
else:
  try:
    s = input('\nEnter space-separated listing of image indices: ')
    indices = [int(index) for index in s.split()]
  except Exception as e:
    print('Something went wrong reading your response:')
    print(e)
    sys.exit(-1)

# Tell user what we're doing
print('OK! Will generate a JSON file including the MNIST data for the images at:')
print(indices)

# I could write my own reader here, but let's just use the tensorflow code
from tensorflow.examples.tutorials.mnist import input_data

mnist_filename='../webapp/models/mnist-model.meta'
data_filename = '../webapp/models/MNIST_data'
mnist = input_data.read_data_sets(data_filename, one_hot=True)

print('  Loaded MNIST data...')
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph(mnist_filename)
new_saver.restore(sess, tf.train.latest_checkpoint('../webapp/models'))
x = tf.get_collection('mnist')[0]
F = tf.get_collection('mnist')[3]
print('  Loaded MNIST classifier...')
print('  Starting to process images...')

# Create directory for images
if not os.path.exists('imgs'):
  os.makedirs('imgs')

# Create dictionary
data_to_write = {}
for num, curr_index in enumerate(indices):
  curr_image = mnist.test._images[curr_index]
  curr_label = F.eval(feed_dict={x:curr_image.reshape((-1,784))})[0] # mnist.test._labels[curr_index]
  curr_class = argmax(curr_label)

  if from_file:
    display_name = display_names[num]
  else:
    display_name = '{} ({})'.format(curr_class, curr_index)

  data_to_write[display_name] = {
      'image':curr_image.tolist(),
      'likelihoods':curr_label.tolist(),
      'class':str(curr_class),
      'MNIST_ID':str(curr_index)
    }

  print('    Processed image {} (class={}; image {} of {})'.format(curr_index, curr_class, num+1, len(indices)))

  imsave('imgs/{}.png'.format(display_name), curr_image.reshape((28,28)))
  

print('  Finished all images...')

filename = 'seeds.json'
with open(filename, 'w') as f:
  json.dump(data_to_write, f)

print('Wrote data to {}'.format(filename))
