import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch = 100

X = tf.placeholder(tf.float32 , [None , 784])
X_reshaped = tf.reshape(X , [-1,28,28,1])
Y = tf.placeholder(tf.float32 , [None , 10])

mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)


def trainown(X,kernel,kernel2,kernel3,kernel4,kernel5,droupout_prob):

	ksize = [1,2,2,1]
	strides = [1,1,1,1]
	strides2 = [1,2,2,1]

	#layer 1
	filter1 = tf.Variable(tf.random_normal(kernel , stddev = 0.05))
	layer1_conv = tf.nn.relu(tf.nn.conv2d(X ,filter1 , strides = strides , padding = 'SAME'))
	layer1_maxpool = tf.nn.relu(tf.nn.max_pool(layer1_conv , ksize = ksize , strides = [1,2,2,1] , padding = 'SAME'))

	#layer 2
	filter2 = tf.Variable(tf.random_normal(kernel2 , stddev = 0.05))
	layer2_conv = tf.nn.relu(tf.nn.conv2d(layer1_maxpool ,filter2 ,strides = strides , padding = 'SAME'))
	layer2_maxpool = tf.nn.relu(tf.nn.max_pool(layer2_conv , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer2 = tf.nn.dropout(layer2_maxpool , droupout_prob)

	#layer3
	filter3 = tf.Variable(tf.random_normal(kernel3 , stddev = 0.05))
	layer3_conv = tf.nn.relu(tf.nn.conv2d(layer2 , filter3 ,strides = strides , padding ='SAME'))
	layer3_maxpool = tf.nn.relu(tf.nn.max_pool(layer3_conv , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer3 = tf.nn.dropout(layer3_maxpool , droupout_prob)

	#layer4-First fully connected layer.
	filter4 = tf.Variable(tf.random_normal(kernel4 , stddev = 0.05))
	shapestraight = [-1 , 4*4*128]
	layer4_shaping = tf.reshape(layer3 , shapestraight)
	layer4drop = tf.nn.dropout(layer4_shaping , droupout_prob)
	layer4 = tf.nn.relu(tf.matmul(layer4drop , filter4))

	#layer5-Output layer
	filter5 = tf.Variable(tf.random_normal(kernel5 ,stddev = 0.05))
	layer5drop = tf.nn.dropout(layer4 , droupout_prob)
	output = tf.nn.relu(tf.matmul(layer5drop , filter5))

	return output

kernel = [3,3,1,32]
kernel2 = [3,3,32,64]
kernel3 = [3,3,64,128]
kernel4 = [4 * 4 * 128,500]
kernel5 = [500 , 10]

droupout_prob = tf.placeholder(tf.float32)

train_layer = trainown(X_reshaped,kernel,kernel2,kernel3,kernel4,kernel5,droupout_prob)

Y_ = tf.nn.softmax(train_layer)
correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

cost = tf.nn.softmax_cross_entropy_with_logits(logits = train_layer , labels = Y)
optimize = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(init)

	for ep in range(10):

		for i in range(int(len(mnist.train.labels)/batch)):
			trainimg , trainlbl = mnist.train.next_batch(batch_size = batch)
			data = {X : trainimg , Y: trainlbl , droupout_prob: 0.5}

			session.run(optimize , feed_dict = data)

		#data = {X : mnist.test.images , Y : mnist.test.labels , droupout_prob : 1}

		#a,c = session.run([accuracy , cost] , feed_dict = data)

		#print("accuracy = ",(a))
	data = {X : mnist.test.images , Y : mnist.test.labels , droupout_prob : 1}

	print("after training : ")
	print(session.run(accuracy , feed_dict = data))

	












