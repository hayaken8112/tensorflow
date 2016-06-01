import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#placeholder
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

# cost_function
with tf.name_scope("softmax") as scope:
    y = tf.nn.softmax(tf.matmul(x,W) + b)

# cross_entropy
with tf.name_scope("xent") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
    

# model_training
with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# summary
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_beginners_logs", sess.graph_def)
tf.initialize_all_variables().run()

for i in range(1000):
    if i % 10 == 0:
        feed = {x: mnist.test.images, y_: mnist.test.labels}
        result = sess.run([merged, accuracy], feed_dict=feed)
        summay_str = result[0]
        acc = result[1]
        writer.add_summary(summay_str, i)
        print("Accuracy at step %s: %s" %(i, acc))
    else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed = {x: batch_xs, y_: batch_ys}
        sess.run(train_step, feed_dict=feed)
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
