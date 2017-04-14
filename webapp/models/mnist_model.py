import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial=tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
  
def train_and_save_model(filename):
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
  # Placeholders:
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  # Model Parameters
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1, 28, 28, 1]) # x is a [picture_ct, 28*28], so x_image is [picture_ct, 28, 28, 1]
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  W_fc1 = weight_variable([7*7*64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  keep_prob = tf.placeholder_with_default(1.0, ())
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2=weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_count = tf.count_nonzero(correct_prediction)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  

  # Set up training criterion
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  
  # Initializer step
  init_op = tf.global_variables_initializer() # must be after adamoptimizer, since that creates more vars
  
  # Configure saver
  saver = tf.train.Saver()
  tf.add_to_collection('mnist', x)
  tf.add_to_collection('mnist', y_)
  tf.add_to_collection('mnist', keep_prob)
  tf.add_to_collection('mnist', y_conv)
  tf.add_to_collection('mnist', correct_count)
  tf.add_to_collection('mnist', cross_entropy)

  
  # Train the model
  with tf.Session() as sess:
    sess.run(init_op)
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
          print("Step {}: Training accuracy {}".format(i, train_accuracy))
      sess.run(train_step, feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})
      
    save_path = saver.save(sess, filename)
    print('Model saved to: {}'.format(filename))
  
if __name__ == '__main__':
  train_and_save_model('./mnist-model')
