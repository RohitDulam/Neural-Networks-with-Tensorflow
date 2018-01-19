import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


batch = 60
mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)
X_shape = tf.placeholder(tf.float32 , [None , 784])
X = tf.reshape(X_shape , [-1,28,28,1])

def noise(X):

	X_noise = tf.multiply(X , tf.random_uniform(shape = tf.shape(X) , minval = 0 , maxval = 3 , dtype = tf.float32))

	return X_noise

def leakyrelu(tnsr, alpha = 0.15):
	return tf.maximum(alpha*tnsr , tnsr)

def Encoder(x):
	stride = [1,1,1,1]
	ksize = [1,2,2,1]
	maxpool = [1,2,2,1]

	filter1 = tf.Variable(tf.random_normal([3,3,1,32] , stddev = 0.005))
	l1_conv = leakyrelu(tf.nn.conv2d(x , filter1 , strides = stride , padding = 'SAME'))
	l1_pool = tf.nn.max_pool(l1_conv , ksize = ksize , strides = maxpool , padding = 'SAME')
	l1 = leakyrelu(l1_pool)
	#print(l1.get_shape().as_list())

	filter2 = tf.Variable(tf.random_normal([3,3,32,32] , stddev = 0.005))
	l2_conv = leakyrelu(tf.nn.conv2d(l1 , filter2 , strides = stride , padding = 'SAME'))
	l2_pool = tf.nn.max_pool(l2_conv , ksize = ksize , strides = maxpool , padding = 'SAME')
	l2 = leakyrelu(l2_pool)

	return l2

def Decoder(y):

	stride = [1,1,1,1]

	filter_1 = tf.Variable(tf.random_normal([3,3,32,32] , stddev = 0.005))
	l1_conv = leakyrelu(tf.nn.conv2d(y , filter_1 , strides = stride , padding = 'SAME'))

	filter_2 = tf.Variable(tf.random_normal([3,3,32,32] , stddev = 0.005))
	l2 = tf.nn.conv2d_transpose(l1_conv , filter_2 , output_shape = [batch , 14 , 14 , 32] , strides = [1,2,2,1] , padding = 'SAME')

	filter_3 = tf.Variable(tf.random_normal([3,3,32,32] , stddev = 0.005))
	l3 = tf.nn.conv2d_transpose(l2 , filter_2 , output_shape = [batch , 28 , 28 , 32] , strides = [1,2,2,1] , padding = 'SAME')

	filter_4 = tf.Variable(tf.random_normal([3,3,32,1] , stddev = 0.005))
	l4 = leakyrelu(tf.nn.conv2d(l3 , filter_4 , strides = stride , padding = 'SAME'))

	logits = tf.nn.sigmoid(l4)

	return logits


Logits = Decoder(Encoder(noise(X)))

correct = tf.equal(tf.argmax(X,1) , tf.argmax(Logits , 1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

#cost = tf.nn.softmax_cross_entropy_with_logits(logits = train_layer , labels = Y)


cost = tf.nn.softmax_cross_entropy_with_logits(logits = Logits , labels = X)
optimize = tf.train.AdamOptimizer().minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as session:

	session.run(init)

	for it in range(10):

		for i in range(int(len(mnist.train.labels)/batch)):

			X_mini , _ = mnist.train.next_batch(batch_size = batch)

			data = {X_shape : X_mini}

			session.run(optimize , feed_dict = data)


	print('After training : ')
	print(session.run(accuracy , feed_dict = {X_shape : mnist.test.images}))


	


