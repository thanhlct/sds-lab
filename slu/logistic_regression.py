from __future__ import print_function, division
import tensorflow as tf
import numpy as np
#import input_data
#import data_manager as input_data
import data_uni_manager as input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w)  # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("./data", 50.0)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
vlX, vlY = mnist.valid.images, mnist.valid.labels

#x_size = 858#100% bigram
#x_size = 344#40% bigram
x_size = 146#50% unigram
y_size =2

X = tf.placeholder("float", [None, x_size])  # create symbolic variables
Y = tf.placeholder("float", [None, y_size])

w = init_weights([x_size, y_size])  # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))  # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # construct optimizer
predict_op = tf.argmax(py_x, 1)  # at predict time, evaluate the argmax of the logistic regression

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(801):   
    if i%50==0:
        train_predict = sess.run(predict_op, {X:trX})
        valid_predict = sess.run(predict_op, {X:vlX})
        test_predict = sess.run(predict_op, {X:teX})
        train_acc = np.mean(np.argmax(trY, 1)==train_predict)
        valid_acc = np.mean(np.argmax(vlY, 1)==valid_predict)
        test_acc = np.mean(np.argmax(teY, 1)==test_predict)
        print('%d\t%.3f\t%.3f\t%.3f'%(i, train_acc, valid_acc, test_acc))

    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    #print(i, np.mean(np.argmax(teY, axis=1)==sess.run(predict_op, feed_dict={X: teX, Y: teY})))

