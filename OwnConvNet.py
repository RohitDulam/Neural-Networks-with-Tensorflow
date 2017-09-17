import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch_size = 100

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

trainX,trainY,testX,testY = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels 

X = tf.placeholder(tf.float32 , [None,28,28,1])
Y = tf.placeholder(tf.float32 , [None,10])

trainX = trainX.reshape(-1,28,28,1)
testX = testX.reshape(-1,28,28,1)

weights = tf.Variable(tf.random_normal([7,7,1,1] , stddev = 0.1))
weight_out = tf.Variable(tf.random_normal([196,10] , stddev = 0.01))

def train(X,weights,weight_out):
	lconv = tf.nn.relu(tf.nn.conv2d(X,weights,strides = [1,1,1,1], padding = 'SAME'))
	lmax = tf.nn.relu(tf.nn.max_pool(lconv,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME'))
	lmax = tf.reshape(lmax ,[-1 , 196])
	lout = tf.nn.relu(tf.matmul(lmax,weight_out))
	return lout 

out = train(X,weights,weight_out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out , labels = Y ))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(out,1)

with tf.Session() as session:
	tf.global_variables_initializer().run()


	session.run(train_op, feed_dict = {X : trainX[:batch_size], Y : trainY[:batch_size] })


	

	print(np.mean(session.run(predict_op , feed_dict = {X : testX[:batch_size], Y : testY[:batch_size] })))



