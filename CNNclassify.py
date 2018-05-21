import os
import cv2
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.interactive(False)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.datasets import cifar10

# Class names which are given in CIFAR-10 dataset
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Load the dataset as training set and testing set
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Normalize to zero mean
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xTest -= meanImage

# Set the data batch variables
batchSize = 256
epochs = 200
printEvery = 10
keep_prob = tf.placeholder("float")

tf.reset_default_graph()

# initialize input data shape and datatype for data and labels
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input")
y = tf.placeholder(tf.int64, [None])

# Convolutional layer 1 with relu  --------------> filter size = 5*5 , stride = 1 , number of filters = 32
wC1 = tf.get_variable("wC1", shape=[5, 5, 3, 32])
bC1 = tf.get_variable("bC1", shape=[32])
C1 = tf.nn.conv2d(x, wC1, strides=[1, 1, 1, 1], padding='SAME') + bC1
C1_relu = tf.nn.relu(C1)
# 2*2 Max pooling
max_pool_C1 = tf.nn.max_pool(value=C1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional layer 2 with relu --------------> filter size = 5*5 , stride = 1 , number of filters = 64
wC2 = tf.get_variable("wC2", shape=[5, 5, 32, 64])
bC2 = tf.get_variable("bC2", shape=[64])
C2 = tf.nn.conv2d(max_pool_C1, wC2, strides=[1, 1, 1, 1], padding='SAME') + bC2
C2_relu = tf.nn.relu(C2)
# 2*2 Max pooling
max_pool_C2 = tf.nn.max_pool(value=C2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional layer 3 with relu--------------> filter size = 5*5 , stride = 1 , number of filters = 64
wC3 = tf.get_variable("wC3", shape=[5, 5, 64, 64])
bC3 = tf.get_variable("bC3", shape=[64])
C3 = tf.nn.conv2d(max_pool_C2, wC3, strides=[1, 1, 1, 1], padding='SAME') + bC3
C3_relu = tf.nn.relu(C3)
# 2*2 Max pooling
max_pool_C3 = tf.nn.max_pool(value=C3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional layer 4 with relu--------------> filter size = 5*5 , stride = 1 , number of filters = 64
wC4 = tf.get_variable("wC4", shape=[5, 5, 64, 64])
bC4 = tf.get_variable("bC4", shape=[64])
C4 = tf.nn.conv2d(max_pool_C3, wC4, strides=[1, 1, 1, 1], padding='SAME') + bC4
C4_relu = tf.nn.relu(C4)
# 2*2 Max pooling
max_pool_C4 = tf.nn.max_pool(value=C4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

shape = max_pool_C4.get_shape().as_list()
reshape = tf.reshape(max_pool_C4, [-1, shape[1] * shape[2] * shape[3]])

# Fully connected layer 1
w_FC1 = tf.get_variable("w_FC1", shape=[shape[1] * shape[2] * shape[3], 1024])
b_FC1 = tf.get_variable("b_FC1", shape=[1024])
FC1_output = tf.matmul(reshape, w_FC1) + b_FC1
FC1_relu = tf.nn.relu(FC1_output)
FC1_reshape = tf.reshape(FC1_relu, [-1, 1024])

# Fully connected layer 2
w_FC2 = tf.get_variable("w_FC2", shape=[1024, 128])
b_FC2 = tf.get_variable("b_FC2", shape=[128])
FC2_output = tf.matmul(FC1_reshape, w_FC2) + b_FC2
FC2_relu = tf.nn.relu(FC2_output)
FC2_reshape = tf.reshape(FC2_relu, [-1, 128])

dropout = tf.nn.dropout(FC2_reshape, 0.5)

# Output layer
w_out = tf.get_variable("w_out", shape=[128, 10])
b_out = tf.get_variable("b_out", shape=[10])
yOut = tf.add(tf.matmul(dropout, w_out), b_out, name="output")

# Loss function
train_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
meanLoss = tf.reduce_mean(train_loss)

# Optimizer
optimizer = tf.train.AdamOptimizer(6e-4)
trainStep = optimizer.minimize(meanLoss)

# Accuracy
correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

def train():

    print("Loop   Train Loss   Train Acc%   Test Loss    Test Acc%")

    s = np.arange(xTrain.shape[0])
    np.random.shuffle(s)

    for e in range(epochs):
        losses = []
        accs = []
        for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
            j = (i * batchSize) % xTrain.shape[0]
            index = s[j:j + batchSize]
            newBatchSize = yTrain[index].shape[0]

            loss, acc, _ = sess.run([meanLoss, accuracy, trainStep],feed_dict={x: xTrain[index, :], y: yTrain[index]})

            losses.append(loss * newBatchSize)
            accs.append(acc * newBatchSize)

        # Calculate train loss
        train_accuracy = np.sum(accs) /(xTrain.shape[0])
        train_loss = np.sum(losses) / xTrain.shape[0]

        # Calculate test loss
        test_loss, test_accuracy = sess.run([meanLoss, accuracy], feed_dict={x: xTest, y: yTest})

        if e % 20 == 0:
            print('%3s' % int(e / 20 + 1), '%9s' % round(train_loss, 4), '%13s' % round(train_accuracy * 100, 4),
                  '%13s' % round(test_loss, 4), '%11s' % round(test_accuracy * 100, 4))

    # Save the model
    saver = tf.train.Saver()
    saver.save(sess, 'model/model')

# test function
def test(temp):

    # Image to be tested
    filename = temp
    # Restore the model
    saver = tf.train.Saver()
    saver.restore(sess, 'model/model')

    tf.get_default_graph().as_graph_def()
    b_con = sess.graph.get_tensor_by_name("output:0")

    # Resize image
    Image = cv2.imread(filename, 1)
    plt.imshow(Image)
    img = cv2.resize(Image, (32, 32))
    resized_image = np.reshape(img, (-1, 32, 32, 3))
    resized_image = resized_image.astype(np.float32)

    # output prediction for image
    classification = sess.run(b_con, feed_dict={x: resized_image})

    max = sess.run(tf.argmax(classification, 1))
    m = max[0]
    cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(cifar10classes[m])

    # Visualization of the first convolutional layer
    units = sess.run(C1, feed_dict={x: resized_image})

    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))

    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.imshow(units[0, :, :, i], interpolation="None", cmap="gray")

    plt.savefig('CONV_rslt')
    plt.show()

# Initialize tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# # Train and test the model
if sys.argv[1] == "train":
     train()
else:
    test(sys.argv[2])
