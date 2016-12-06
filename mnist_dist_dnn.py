from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

# Parameters
learning_rate = 0.01
regularization_rate = 0.01
training_epochs = 20
batch_size = 100
display_step = 100
keep_prob = 0.5
total_steps = 20000
# Network Parameters
n_input = 784 # Number of feature
n_hidden_1 = 700 # 1st layer number of features
n_hidden_2 = 600 # 2nd layer number of features
n_hidden_3 = 500 # 3nd layer number of features
n_hidden_4 = 400 # 4nd layer number of features
n_hidden_5 = 300 # 5nd layer number of features
n_hidden_6 = 200 # 6nd layer number of features
n_hidden_7 = 100 # 7nd layer number of features
n_hidden_8 = 50 # 8nd layer number of features
n_classes = 10 # Number of classes to predict


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    # Hidden layer with RELU activation
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.relu(layer_6)
    # Hidden layer with RELU activation
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.relu(layer_7)
    # Hidden layer with RELU activation
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.relu(layer_8)
    # Drop-out layer
    # layer_3_drop = tf.nn.dropout(layer_2, keep_prob)
    # Output layer with linear activation
    # out_layer = tf.add(tf.matmul(layer_3_drop, weights['out']), biases['out'])
    out_layer = tf.add(tf.matmul(layer_8, weights['out']), biases['out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def regularization_value(weights, biases):
    value = regularization_rate * tf.nn.l2_loss(weights['h1']) \
            + regularization_rate * tf.nn.l2_loss(weights['h2']) \
            + regularization_rate * tf.nn.l2_loss(weights['h3']) \
            + regularization_rate * tf.nn.l2_loss(weights['h4']) \
            + regularization_rate * tf.nn.l2_loss(weights['h5']) \
            + regularization_rate * tf.nn.l2_loss(weights['h6']) \
            + regularization_rate * tf.nn.l2_loss(weights['h7']) \
            + regularization_rate * tf.nn.l2_loss(weights['h8']) \
            + regularization_rate * tf.nn.l2_loss(weights['out']) \
            + regularization_rate * tf.nn.l2_loss(biases['b1']) \
            + regularization_rate * tf.nn.l2_loss(biases['b2']) \
            + regularization_rate * tf.nn.l2_loss(biases['b3']) \
            + regularization_rate * tf.nn.l2_loss(biases['b4']) \
            + regularization_rate * tf.nn.l2_loss(biases['b5']) \
            + regularization_rate * tf.nn.l2_loss(biases['b6']) \
            + regularization_rate * tf.nn.l2_loss(biases['b7']) \
            + regularization_rate * tf.nn.l2_loss(biases['b8']) \
            + regularization_rate * tf.nn.l2_loss(biases['out'])
    return value

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Get data ...
            mnist = input_data.read_data_sets("data", one_hot=True)

            # Define variables
            x = tf.placeholder(tf.float32, [None, n_input])
            y_ = tf.placeholder(tf.float32, [None, 10])

            # Create the model...
            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0.0, stddev=1.0/tf.sqrt(n_input*1.0),
                                                   dtype=tf.float32, name='h1_weights')),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_1*1.0),
                                                   dtype=tf.float32, name='h2_weights')),
                'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_2*1.0),
                                                   dtype=tf.float32, name='h3_weights')),
                'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_3*1.0),
                                                   dtype=tf.float32, name='h4_weights')),
                'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_4*1.0),
                                                   dtype=tf.float32, name='h5_weights')),
                'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_5*1.0),
                                                   dtype=tf.float32, name='h6_weights')),
                'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_6*1.0),
                                                   dtype=tf.float32, name='h7_weights')),
                'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_7*1.0),
                                                   dtype=tf.float32, name='h8_weights')),
                'out': tf.Variable(tf.random_normal([n_hidden_8, n_classes], mean=0.0, stddev=1.0/tf.sqrt(n_hidden_8*1.0),
                                                    dtype=tf.float32, name='out_weights'))
            }
            biases = {
                # 'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                # 'out': tf.Variable(tf.random_normal([n_classes]))
                'b1': tf.Variable(tf.zeros([n_hidden_1])),
                'b2': tf.Variable(tf.zeros([n_hidden_2])),
                'b3': tf.Variable(tf.zeros([n_hidden_3])),
                'b4': tf.Variable(tf.zeros([n_hidden_4])),
                'b5': tf.Variable(tf.zeros([n_hidden_5])),
                'b6': tf.Variable(tf.zeros([n_hidden_6])),
                'b7': tf.Variable(tf.zeros([n_hidden_7])),
                'b8': tf.Variable(tf.zeros([n_hidden_8])),
                'out': tf.Variable(tf.zeros([n_classes]))
            }
            pred = multilayer_perceptron(x, weights, biases)

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_)
                                  + regularization_value(weights, biases))

            global_step = tf.Variable(0)

            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/Users/urey/PycharmProjects/tensorflow_demo/notes/tensorflow/checkpoint/",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 2000 steps have completed).
        step = 0
        while not sv.should_stop() and step < total_steps:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, loss_v, step = sess.run([train_op, cost, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            if step % display_step == 0:
                print("Step %d in task %d, loss %f" % (step, FLAGS.task_index, loss_v))
        print("done.")

        if FLAGS.task_index != 0:
            print("accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels}))


if __name__ == "__main__":
    tf.app.run()

