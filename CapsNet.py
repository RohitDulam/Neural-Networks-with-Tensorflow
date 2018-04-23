import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


def squash(tensr , axis = -1 , epsilon = 1e-7):
    sq_norm = tf.reduce_sum(tf.square(tensr) , axis = axis , keep_dims = True)
    norm = tf.sqrt(sq_norm + epsilon)
	# squashing formula = (tensr^2 / (1 + tensr^2)) * (tensr / norm)
	# Squash Factor
    part_a = sq_norm / (1 + sq_norm)
	#Unit Vector
    part_b = tensr / norm
    return part_a * part_b

batch = 64
X_shape = tf.placeholder(tf.float32 , [None , 784])
X = tf.reshape(X_shape , [batch,28,28,1])
#y = tf.placeholder(shape=[None , 10], dtype=tf.int64)
y = tf.placeholder(shape = [None] , dtype = tf.int64)

kernels = tf.Variable(tf.random_normal([9 , 9 , 1 , 256] , stddev = 0.005))
l1 = tf.nn.relu(tf.nn.conv2d(X, kernels , strides = [1,1,1,1] , padding = 'VALID'))

kernel2 = tf.Variable(tf.random_normal([9 , 9 , 256 , 256] , stddev = 0.005))
l2 = tf.nn.conv2d(l1, kernel2 , strides = [1,2,2,1] , padding = 'VALID')
l2_reshape = tf.reshape(l2 , [-1 , 1152 , 8])

caps_1_output = squash(l2_reshape)

layer1_caps = 1152 # 32 * 6 * 6
layer2_caps = 16
layer2_classes = 10
layer1_vector_dimension = 8 

weights_untiled = tf.Variable(tf.random_normal([1 , layer1_caps , layer2_classes , layer2_caps , layer1_vector_dimension] , stddev = 0.005))

weights = tf.tile(weights_untiled , [batch, 1, 1, 1, 1])

caps_1_expand_1 = tf.expand_dims(caps_1_output , -1)
caps_1_expand_2 = tf.expand_dims(caps_1_expand_1 , 2)

caps_1 = tf.tile(caps_1_expand_2 , [1, 1, 10, 1, 1])
caps_2_predicted = tf.matmul(weights , caps_1)
raw_weights = tf.zeros([batch , 1152 , 10 , 1 , 1] , dtype = tf.float32) # b in the paper. Updated by adding it constantly to the agreement
routing_weights = tf.nn.softmax(raw_weights , dim = 2) #softmax(b) : Measures how likely capsule i may activate capsule j
predictions = tf.multiply(routing_weights , caps_2_predicted) #multiplication of candidate values and predictions of capsule i in higher layer capsule j
predictions_sum = tf.reduce_sum(predictions , 1 , keep_dims = True)
caps_2_round_1 = squash(predictions_sum , axis = -2) # end of round 1

# beginning of round 2
# First find how much do u_hat_j|i agrees with v_i_j

caps_2_round_1_tile = tf.tile(caps_2_round_1 , [1, 1152, 1, 1, 1])
agreement = tf.matmul(caps_2_predicted , caps_2_round_1_tile , transpose_a = True)
raw_weights = raw_weights + agreement
routing_weights_2 = tf.nn.softmax(raw_weights , dim = 2)
predictions_2 = tf.multiply(routing_weights_2 , caps_2_predicted)
predictions_sum_2 = tf.reduce_sum(predictions_2 ,axis = 1 , keep_dims = True)
caps_2_round_2 = squash(predictions_sum_2 , axis = -2)

# Trying to create a dynamic loop

def condition(input, counter):
    return tf.less(counter, 3)

def loop_body(input, counter):
	if counter == 1:
		global raw_weights
	caps_tile = tf.tile(input , [1, 1152, 1, 1, 1])
	agreement = tf.matmul(caps_2_predicted , caps_tile , transpose_a = True)
	raw_weights = raw_weights + agreement
	routing_weights = tf.nn.softmax(raw_weights , dim = 2)
	predictions = tf.multiply(routing_weights , caps_2_predicted)
	predictions_sum = tf.reduce_sum(predictions ,axis = 1 , keep_dims = True)
	caps_final_output = squash(predictions_sum , axis = -2)
	return caps_final_output , tf.add(counter , 1)

counter = tf.constant(1)

result = tf.while_loop(condition, loop_body, [caps_2_round_2, counter])
    
def safe_norm(tnsr, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    squared_norm = tf.reduce_sum(tf.square(tnsr), axis=axis,
                                     keep_dims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)

y_inter = safe_norm(result[0], axis = -2)
y_max = tf.argmax(y_inter , axis = 2)
y_pred = tf.squeeze(y_max, axis=[1,2])
#y_pred = tf.one_hot(y_pred_test , depth = 10)

# The following code illustrates the margin loss.

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y , depth = 10)
#T = tf.reshape(T_ , shape = (-1 , 10))
result_norm = safe_norm(result[0] , axis = -2 , keep_dims = True)
present_error_inter = tf.square(tf.maximum(0. , m_plus - result_norm))
present_error = tf.reshape(present_error_inter , shape = (-1 , 10))

absent_error_inter = tf.square(tf.maximum(0. , result_norm - m_minus))
absent_error = tf.reshape(absent_error_inter , shape = (-1 , 10))

Loss = tf.add(T * present_error , lambda_ * (1.0 - T) * absent_error)
margin_loss = tf.reduce_mean(tf.reduce_sum(Loss , axis = 1))

# Masking the input for training, this is the part i'm not totally clear about.

mask = tf.placeholder_with_default(False , shape = ())
recon_targets = tf.cond(mask , lambda: y , lambda: y_pred)
reconstruction = tf.one_hot(recon_targets , depth = 10)
recon_reshape = tf.reshape(reconstruction , [-1 , 1 , 10 , 1 , 1])

masked_ouput = tf.multiply(result[0] , recon_reshape)
decoder_input = tf.reshape(masked_ouput , [-1 , 160])

#decoder network which is plain feed forward neural network

weights_1 = tf.Variable(tf.random_normal([160 , 512] , stddev = 0.005))
bias_1 = tf.Variable(tf.random_normal( [512] , stddev = 0.005))
weights_2 = tf.Variable(tf.random_normal([512 , 1024] , stddev = 0.005))
bias_2 = tf.Variable(tf.random_normal( [1024] , stddev = 0.005))
weights_3 = tf.Variable(tf.random_normal([1024 , 784] , stddev = 0.005))
bias_3 = tf.Variable(tf.random_normal( [784] , stddev = 0.005))

l1 = tf.nn.relu(tf.add(tf.matmul(decoder_input , weights_1) , bias_1))
l2 = tf.nn.relu(tf.add(tf.matmul(l1 , weights_2) , bias_2))
decoder_output = tf.nn.sigmoid(tf.add(tf.matmul(l2 , weights_3) , bias_3))

flat_x = tf.reshape(X , [-1 , 784])
difference = tf.square(flat_x - decoder_output)
reconstruction_loss = tf.reduce_mean(difference)

# final loss
alpha = 0.0005
total_loss = tf.add(margin_loss , alpha * reconstruction_loss)

correct = tf.equal(y , y_pred)
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))
training = tf.train.AdamOptimizer().minimize(total_loss)

init = tf.global_variables_initializer()
n_epochs = 2

n_iterations_per_epoch = mnist.train.num_examples // batch
n_iterations_validation = mnist.validation.num_examples // batch

with tf.Session() as session:
	init.run()

	for _ in range(n_epochs):
		for i in range(int(len(mnist.train.labels)/batch)):
			train_x , train_y = mnist.train.next_batch(batch_size = batch)
			session.run([result] , feed_dict = {X_shape : train_x})
			data = {X_shape : train_x , y : train_y , mask : True}
			session.run([training] , feed_dict = data)

	acc = []
	loss = []
	for i in range(int(len(mnist.test.labels)/batch)):
		X_batch, Y_batch = mnist.test.next_batch(batch_size = batch)
		data = {X_shape : X_batch , y : Y_batch}
		a , l = session.run([accuracy , total_loss] , feed_dict = data)
		acc.append(a)
		loss.append(l)

	acc_testing = np.mean(acc) * 100
	loss_testing = np.mean(loss)
	print("After Training :")
	print(acc_testing)

        






















