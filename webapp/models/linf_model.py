# ML includes
import tensorflow as tf
import numpy as np
import pandas as pd

# General python includes
import os
import math
import json
from itertools import permutations

# Altered from Cleverhans
# (Github: https://github.com/openai/cleverhans)
def fgsm(seed_image, seed_class, epsilon):
    """
    TensorFlow implementation of the Fast Gradient
    Sign method.
    :param x: the input placeholder
    :param F: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    Taken from cleverhans and modified.
    """
    x         = tf.get_collection('mnist')[0]
    y_        = tf.get_collection('mnist')[1]
    keep_prob = tf.get_collection('mnist')[2]
    F         = tf.get_collection('mnist')[3]
    loss      = tf.get_collection('mnist')[5]
    epsilon   = float(epsilon)
    orig = seed_image
    
    # Compute loss
    y = tf.to_float(tf.equal(F, tf.reduce_max(F, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = epsilon * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)
    
    adv_x = tf.clip_by_value(adv_x, 0, 1) # clip to [0,1]

    normal_classification = np.zeros((1,10))
    normal_classification[0,int(seed_class)] = 1
    adv = adv_x.eval(feed_dict={x:orig, keep_prob:1.0, y_:normal_classification})
    
    ### Create plot of relative likelihoods for each class ###
    adv_probs  = F.eval(feed_dict={x:adv, keep_prob:1.0})[0]
    norm_probs = F.eval(feed_dict={x:orig, keep_prob:1.0})[0]

    adv_scaled  = (adv_probs - adv_probs.min()) / adv_probs.ptp()
    norm_scaled = (norm_probs - norm_probs.min()) / norm_probs.ptp()
    
    return adv_probs.argmax(), np.reshape(adv, (28, 28)), adv_probs
    
def setup(mnist_filename):    
  # Will run on import
  print('Setting up the L1 model with MNIST model at {}'.format(mnist_filename))
  sess = tf.InteractiveSession()
  new_saver = tf.train.import_meta_graph(mnist_filename)
  new_saver.restore(sess, tf.train.latest_checkpoint('./webapp/models'))
