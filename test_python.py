import numpy as np
import tensorflow as tf


a = tf.Variable(tf.random_normal([10, 3], mean=0.0, stddev=1.0/tf.sqrt(10*1.0),
                                                   dtype=tf.float32, name='h1_weights'))
print a
