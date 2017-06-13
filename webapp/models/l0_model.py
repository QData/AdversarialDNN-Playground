# ML includes
import tensorflow as tf
import numpy as np
import pandas as pd

# General python includes
import os
import math
import json
from itertools import permutations, combinations, product

# Plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')

def grad(F):
  x = tf.get_collection('mnist')[0]
  grad_list = [ tf.gradients(F[:,i], x)[0] for i in range(10) ] # List of gradient tensors
  return tf.stack(grad_list, axis=2) # dimension = (?, 784, 10)

def slow_map(grad_F, X, t, feature_set):
  print('using slow map')
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

# Uses the numpy functions for argmax like cleverhans
def faster_map(grad_F, x_adversary, t, feature_set):
  print('Using fastest map')
  x = tf.get_collection('mnist')[0]

  gF = grad_F.eval(feed_dict = {x:x_adversary}).squeeze()
  num_raw_features = gF.shape[0]
  target_vector = gF[:, t].reshape(num_raw_features)
  other_vector  = np.sum(gF, axis=1).reshape(num_raw_features) - target_vector # Get sum of "all but target col"
  
  k = int(len(feature_set)*.25) # consider the top quarter of the feature set; magic number to match cleverhans
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
    
# Original FJSMA implementation; does not use numpy to fullest advantage
# Not used; preserved "just in case."  Remove later.
def fast_map(grad_F, x_adversary, t, feature_set):
  print('Using fast map')
  x = tf.get_collection('mnist')[0]
  M = 0 # Max -alpha*beta
  p1 = None
  p2 = None
  M_nolimits = 0
  p1_nolimits = None
  p2_nolimits = None
  
  gF = grad_F.eval(feed_dict = {x:x_adversary})
  pixelSumGF = np.sum(gF, axis=2) # sum along the innermost axis

  top_ct  = int(len(feature_set)*.1) # consider the top tenth of the feature set
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
      
def attack(X, target_class, max_distortion, fast=False):
  # unpack the string parameters into non-string parameters, as needed
  max_distortion = float(max_distortion)
  
  # Get the feed dict parameters we needed
  x = tf.get_collection('mnist')[0]
  keep_prob = tf.get_collection('mnist')[2]
  
  orig = np.copy(X)
  F = tf.get_collection('mnist')[3]

  feature_set = {i for i in range(X.shape[1]) if X[0, i] != 0}
  curr_iter = 0
  max_iter = math.floor(784*max_distortion / 2)

  classify_op = tf.argmax(F,1)
  gradF = grad(F)
  
  source_class = classify_op.eval(feed_dict={x:X, keep_prob:1.0})
  print('Evaled first thing')
  
  saliency_map = faster_map if fast else slow_map
  while source_class != target_class and feature_set and curr_iter < max_iter:
    p1, p2 = saliency_map(gradF, X, target_class, feature_set)
    
    X[0, p1] = max(X[0, p1] - 1, 0)
    X[0, p2] = max(X[0, p2] - 1, 0)
    if X[0, p1] == 0:
      feature_set.remove(p1)
    if X[0, p2] == 0:
      feature_set.remove(p2)
    source_class = classify_op.eval(feed_dict={x:X, keep_prob:1.0})
    curr_iter += 1
    if (curr_iter % 10 == 0):
      print(curr_iter)
  print('Finished {} iterations.'.format(curr_iter))
  
  ### Create plot of relative likelihoods for each class ###
  adv_probs  = F.eval(feed_dict={x:X})[0]
  norm_probs = F.eval(feed_dict={x:orig})[0]
  
  adv_scaled  = (adv_probs - adv_probs.min()) / adv_probs.ptp()
  norm_scaled = (norm_probs - norm_probs.min()) / norm_probs.ptp()

  return source_class[0], np.reshape(X, (28, 28)), adv_probs

  
mnist_data = None

def setup(mnist_filename):    
  global mnist_data
  # Will run on import
  print('Setting up the L1 model with MNIST model at {}'.format(mnist_filename))
  sess = tf.InteractiveSession()
  new_saver = tf.train.import_meta_graph(mnist_filename)
  new_saver.restore(sess, tf.train.latest_checkpoint('./webapp/models'))
  
  with open('./webapp/models/seeds.json') as f:
    mnist_data = json.load(f)
