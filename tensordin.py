import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch = 100
mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True) 

X = tf.placeholder(tf.float32 , [None , 784])
Y = tf.placeholder(tf.float32 , [None , 10])

Weights = tf.Variable(tf.random_normal([784,10] , stddev = 0.005))
bias = tf.Variable(tf.random_normal([10] , stddev = 0.005))

inter = tf.add(tf.matmul(X,Weights),bias)
Y_ = tf.nn.softmax(inter)

cost = tf.nn.softmax_cross_entropy_with_logits(logits = inter , labels = Y)

correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_ , 1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(init)

	for ep in range(10):

		for i in range(int(len(mnist.train.labels)/batch)):
			trainimg , trainlbl = mnist.train.next_batch(batch_size = batch)
			data = {X : trainimg , Y: trainlbl}

			session.run(optimizer , feed_dict = data)

	data = {X : mnist.test.images , Y : mnist.test.labels}

	print("after training : ")
	print(session.run(accuracy , feed_dict = data))













