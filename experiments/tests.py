print('starting system')
##############################################
#              Includes                      #
##############################################

# General python includes
import os
import math
import csv
import time
import random
import sys
from itertools import product, combinations
from functools import partial 

import argparse

# ML includes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd

# Cleverhans benchmark
from cleverhans.cleverhans import attacks_tf
from tensorflow.python.platform import flags


flags.DEFINE_integer('nb_classes', 10, 'Number of classes')

print('Packages imported...')

##############################################
#              FJSMA                         #
##############################################

def grad(F):
    grad_list = [ tf.gradients(F[:,i], x)[0] for i in range(10) ] # List of gradient tensors
    return tf.stack(grad_list, axis=2) # dimension = (?, 784, 10)
     
def slow_map(grad_F, X, t, feature_set, ignore_param):
  # Get the feed dict parameters we needed
  x = tf.get_collection('mnist')[0]
  keep_prob = tf.get_collection('mnist')[2]
  
  M = 0 # Max -alpha*beta
  p1 = None
  p2 = None
  M_nolimits = 0
  p1_nolimits = None
  p2_nolimits = None

  gF = grad_F.eval(feed_dict = {x:X, keep_prob:1.0})
  pixelSumGF = np.sum(gF, axis=2) # sum along the innermost axis
  
  for (p, q) in combinations(feature_set, 2):
    alpha = gF[:, p, t] + gF[:, q, t]
    beta = pixelSumGF[:,p] + pixelSumGF[:,q] - alpha

    if -alpha*beta > M:
      (p1_nolimits, p2_nolimits) = (p, q)
      M_nolimits = -alpha*beta
      if alpha < 0 and beta > 0:
        (p1, p2) = (p, q)
        M = -alpha*beta

  if p1 is None or p2 is None:
    return p1_nolimits, p2_nolimits
  else:
    return p1, p2

def old_fast_map(grad_F, x_adversary, t, feature_set, k):
    M = 0 # Max -alpha*beta
    p1 = None
    p2 = None
    M_nolimits = 0
    p1_nolimits = None
    p2_nolimits = None
    
    gF = grad_F.eval(feed_dict = {x:x_adversary, keep_prob:1.0})
    pixelSumGF = np.sum(gF, axis=2) # sum along the innermost axis

    top_ct = k # input the size of top-K set # int(len(feature_set)*.1) # consider the top tenth of the feature set
    best_p = sorted(feature_set, key=lambda p: gF[:, p, t])[:top_ct]
    
    for (p, q) in product(best_p, feature_set):
        if p==q:
            continue
            
        alpha = gF[:, p, t] + gF[:, q, t]
        beta = pixelSumGF[:,p] + pixelSumGF[:,q] - alpha
        
        if alpha < 0 and beta > 0 and -alpha*beta > M:
            (p1, p2) = (p, q)
            M = -alpha*beta
        if -alpha*beta > M:
            (p1_nolimits, p2_nolimits) = (p, q)
            M_nolimits = -alpha*beta
    if p1 is None or p2 is None:
        return p1_nolimits, p2_nolimits
    else:
        return p1, p2

def fast_map(grad_F, x_adversary, t, feature_set, k):
    if k == 0:
        return None, None

    gF = grad_F.eval(feed_dict = {x:x_adversary, keep_prob:1.0}).squeeze()
    num_raw_features = gF.shape[0]
    target_vector = gF[:, t].reshape(num_raw_features)
    other_vector  = np.sum(gF, axis=1).reshape(num_raw_features) - target_vector # Get sum of "all but target col"
    
    ordered_feature_set = sorted(feature_set, key=lambda x: target_vector[x])
    best_pixels = ordered_feature_set[:k]
    
    num_features = len(feature_set)
    
    tV_best = target_vector[best_pixels].reshape((1, k))
    oV_best = other_vector[best_pixels].reshape((1, k))
    
    tV_features = target_vector[ordered_feature_set].reshape((num_features, 1))
    oV_features = target_vector[ordered_feature_set].reshape((num_features, 1))
    
    target_sum = tV_best + tV_features
    other_sum  = oV_best + oV_features
    #print(target_sum.shape)
    
    # heavily influenced by cleverhans
    scores = -target_sum * other_sum
    np.fill_diagonal(scores, 0)

    scores_mask = ((target_sum < 0) & (other_sum > 0))
    scores *= scores_mask

    (p1_raw, p2_raw) = np.unravel_index(np.argmax(scores), scores.shape)
    #print('The scores has shape {}'.format(scores.shape))
    p1 = ordered_feature_set[p1_raw]
    p2 = best_pixels[p2_raw]
    if (p1 != p2):
        return p1, p2
    else:
        return None, None
                
def jsma(X, target_class, F, gradF, upsilon_checkpoints, theta, s_map, input_k): 
    X_adversary = np.copy(X)
    
    feature_set = [i for i in range(X.shape[1]) if X[0, i] != 0] # Todo: See if this can be a set
    
    upsilon_to_iter = lambda m_distortion: min(math.floor(784*m_distortion / (2*100)), len(feature_set)/2)
    max_iter = upsilon_to_iter(max(upsilon_checkpoints))
    checkpoints = set(map(upsilon_to_iter, upsilon_checkpoints))

    classify_op = tf.argmax(F,1)   
    source_class = classify_op.eval(feed_dict={x:X, keep_prob:1.0})

    curr_iter = 0
    success   = False
    successpoints = []
    while not success and curr_iter < max_iter:
        p1, p2 = s_map(gradF, X_adversary, target_class, feature_set, int(input_k*len(feature_set)))
        
        if p1 is None or p2 is None:
          break
        
        X_adversary[0, p1] = max(X_adversary[0, p1] - theta, 0)
        X_adversary[0, p2] = max(X_adversary[0, p2] - theta, 0)
        if X_adversary[0, p1] == 0:
            feature_set.remove(p1)
        if X_adversary[0, p2] == 0:
            feature_set.remove(p2)
            
        source_class = classify_op.eval(feed_dict={x:X_adversary, keep_prob:1.0})
        success      = source_class == target_class 
        curr_iter    += 1

        #if curr_iter in checkpoints:
        #    successpoints.append(success[0])

    #successpoints.extend([success[0]]*(len(upsilon_checkpoints) - len(successpoints)))

    return X_adversary, curr_iter, success#, successpoints


##############################################
#              Data download or load         #
##############################################
from tensorflow.examples.tutorials.mnist import input_data

data_filename = '../webapp/models/MNIST_data'
mnist = input_data.read_data_sets(data_filename, one_hot=True)

print('Loaded MNIST data')

##############################################
#              Train or load model           #
##############################################
model_path     = '../webapp/models/'
model_filename = 'mnist-model'
if not os.path.isfile(model_path + model_filename + '.meta'):
  from mnist_model import train_and_save_model
  train_and_save_model(model_path + model_filename)
 
print('Loaded model')

##############################################
#              Session Creation              #
##############################################
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph(model_path + model_filename + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
x             = tf.get_collection('mnist')[0]
y_            = tf.get_collection('mnist')[1]
keep_prob     = tf.get_collection('mnist')[2]
y_conv        = tf.get_collection('mnist')[3]
correct_count = tf.get_collection('mnist')[4]

print('Restored session')

##############################################
#        Actually start the experiment       #
##############################################

output_file      = sys.argv[1]        # output filename
max_distortion   = float(sys.argv[2]) # Upsilon Parameter
checkpoints      = [max_distortion]   # Real upsilon parameter
attack_type      = sys.argv[3]
if attack_type == 'fjsma':
  input_k        = int(sys.argv[4])/100 # consider as percentage of the input size
print('Writing to {} with upsilon={} and attack type {}.'.format(output_file, max_distortion, attack_type))

possible_classes = list(range(10))
num_to_generate  = 10000
field_names      = ['source_class', 
                    'target_class', 
                    'success', 
                    'iterations', 
                    'time']

# if attack_type in ['jsma', 'fjsma']:
#   field_names.extend(['upsilon{}-success'.format(c) for c in checkpoints])

# Write header for csv
with open(output_file, 'w', newline='') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=field_names)
  writer.writeheader()

if attack_type == 'cleverhans':
  gradF = attacks_tf.jacobian_graph(y_conv, x, nb_classes=10)
elif attack_type in ['jsma', 'fjsma']:
  gradF = grad(y_conv)
else:
  print('YIKES, detected type {}'.format(attack_type))
  sys.exit(1)

print('The MNIST test set has {} samples.'.format(mnist.test._num_examples))
print('We will create {} adversarial samples'.format(num_to_generate))

batch_results = []
batch_size    = 50
total_batches = num_to_generate // batch_size

if attack_type == 'fjsma':
  saliency_map = fast_map
elif attack_type == 'jsma':
  saliency_map = slow_map

start_time = time.time()
for i in range(num_to_generate):
  batch  = mnist.test.next_batch(1)
  X      = np.reshape(batch[0][0], (1, 784))
  y_real = batch[1][0].argmax()
  
  # Find target class
  y_tgt  = random.randrange(10)
  while y_tgt == y_real:         # Whoops, select a different target!
    y_tgt = random.randrange(10)
  
  data_to_write = {
    'source_class' : y_real,
    'target_class' : y_tgt,
  }

  sample_start_time = time.time()
  if attack_type=='cleverhans':
    X_adv, success, percent_changed = attacks_tf.jsma(sess, x, y_conv, gradF, X, y_tgt, -1, 
                                                         max_distortion/100, 0, 1)
    iters = percent_changed * 784 / 2
  else:
    X_adv, iters, success = jsma(X, y_tgt, y_conv, gradF, checkpoints, 1, saliency_map, input_k)
    # for j, c in enumerate(checkpoints):
    #   data_to_write['upsilon{}-success'.format(c)] = successpoints[j] if j < len(successpoints) else True
     
  sample_stop_time  = time.time()

  data_to_write['success']    = 1 if success else 0
  data_to_write['iterations'] = iters
  data_to_write['time']       = sample_stop_time - sample_start_time
  
  batch_results.append(data_to_write)
  
  if (i+1) % batch_size == 0:
    with open(output_file, 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=field_names)
      writer.writerows(batch_results)
    batch_results = []

    batch_num    = (i + 1)//batch_size
    elapsed_time = time.time() - start_time
    exp_time     = elapsed_time * total_batches / batch_num
    exp_end      = time.localtime(start_time + exp_time)
    print('Wrote batch {} of {} to file. Expected End: {}'.format(
        batch_num, 
        total_batches,
        time.strftime("%X", exp_end)
      ))
  
  
  #print('Actual: {} | Target: {} | Success: {} | Iterations: {}'.format(
  #        y_real, 
  #        y_tgt, 
  #        1 if success else 0,
  #        iters))
