# データの読み込み
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#placeholder
# 入力
x = tf.placeholder("float", shape=[None, 784])

# 正解出力
y_ = tf.placeholder("float", shape=[None, 10])

# 重み
W = tf.Variable(tf.zeros([784, 10]))

# バイアス
b = tf.Variable(tf.zeros([10]))

#変数の初期化
sess.run(tf.initialize_all_variables())
# 重みの初期化関数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアスの初期化関数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 畳み込み層とプーリング層の定義
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 画像の変換
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 畳み込み層1
with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# プーリング層1
with tf.name_scope('pool1') as scope:
    h_pool1 = max_pool_2x2(h_conv1)

# 畳み込み層1
with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# プーリング層2
with tf.name_scope('pool2') as scope:
    h_pool2 = max_pool_2x2(h_conv2)

# 全結合層1
with tf.name_scope('fc1') as scope:
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全結合層2
with tf.name_scope('fc2') as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 交差エントロピー
with tf.name_scope("xent") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
    

# 学習
with tf.name_scope("train") as scope:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
# 変数初期化
tf.initialize_all_variables().run()

# tensorboardのための設定
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_expert_logs", sess.graph_def)

# 10000回学習行う
# 100回ごとに学習状況表示
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # tensorboard
    summary_str = sess.run(merged, feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
    writer.add_summary(summary_str, i)
print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
