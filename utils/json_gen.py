# Utility script to generate the seeds.json file

import sys
import json
from numpy import argmax

help_text = (  'This script generates the "seeds.json" file for use in the ' 
             + 'webapp, \nallowing a developer to select which images they\'d '
             + 'like to include in the\noptions for the user.')
print(help_text)

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

data_filename = '../webapp/models/MNIST_data'
mnist = input_data.read_data_sets(data_filename, one_hot=True)

print('  Loaded MNIST data...')
print('  Starting to process images...')

# Create dictionary
data_to_write = {}
for num, curr_index in enumerate(indices):
  curr_image = mnist.test._images[curr_index]
  curr_label = mnist.test._labels[curr_index]
  curr_class = argmax(curr_label)

  display_name = '{} ({})'.format(curr_class, curr_index)
  data_to_write[display_name] = {
      'image':curr_image.tolist(),
      'likelihoods':curr_label.tolist(),
      'class':str(curr_class),
      'MNIST_ID':str(curr_index)
    }

  print('    Processed image {} (class={}; image {} of {})'.format(curr_index, curr_class, num, len(indices)))

print('  Finished all images...'

filename = 'seeds.json'
with open(filename, 'w') as f:
  json.dump(data_to_write, f)

print('Wrote data to {}'.format(filename))
