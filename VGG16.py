import numpy as np
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

size = 224
batch_size = 10
X = tf.placeholder(tf.float32 , [None,size,size,1])
#X_reshaped = tf.reshape(X , [-1,size,size,1])
Y = tf.placeholder(tf.float32 , [None , 10])
droupout_prob = tf.placeholder(tf.float32)


def one_hot_label(img):
    label=str(img.split('.')[0])
    if 'gossiping' in label:
        ohl=[1,0,0,0,0,0,0,0,0,0]
        return ohl
    elif 'isolation' in label:
        ohl=[0,1,0,0,0,0,0,0,0,0]
        return ohl
    elif 'laughing' in label:
        ohl=[0,0,1,0,0,0,0,0,0,0]
        return ohl
    elif 'lp' in label or 'pullinghair' in label:
        ohl=[0,0,0,1,0,0,0,0,0,0]
        return ohl
    elif 'punching' in label:
        ohl=[0,0,0,0,1,0,0,0,0,0]
        return ohl
    elif 'slapping' in label:
        ohl=[0,0,0,0,0,1,0,0,0,0]
        return ohl
    elif 'stabbing' in label:
        ohl=[0,0,0,0,0,0,1,0,0,0]
        return ohl
    elif 'strangle' in label:
        ohl=[0,0,0,0,0,0,0,1,0,0]
        return ohl
    elif '00' in label:
        ohl=[0,0,0,0,0,0,0,0,1,0]
        #print(label)
        return ohl
    else:
        ohl=[0,0,0,0,0,0,0,0,0,1]
        return ohl

def train_data_with_label():
    train_images=[]
    #print("hi")
    for i in tqdm(os.listdir(train_data)):
        path=os.path.join(train_data,i)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        train_images.append([np.array(img),one_hot_label(i)])
    shuffle(train_images)
    #print("hi")
    #print("\nTraining images:",len(train_images))
    return train_images


def test_data_with_label():
    test_images=[]
    for i in tqdm(os.listdir(test_data)):
        path=os.path.join(test_data,i)
        #print(path)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        test_images.append([np.array(img),one_hot_label(i)])
        shuffle(test_images)
    return test_images


training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,size,size,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,size,size,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

def vgg(X,kernel,kernel_,kernel2,kernel2_,kernel3,kernel3_,kernel4,kernel5,kernel6,kernel7,kernel8,droupout_prob):

	ksize = [1,2,2,1]
	strides = [1,1,1,1]
	strides2 = [1,2,2,1]
	X /= 255

	#CONVOLUTION BLOCK - 1
	filter1 = tf.Variable(tf.random_normal(kernel , stddev = 0.05))
	layer1_conv_1 = tf.nn.relu(tf.nn.conv2d(X ,filter1 , strides = strides , padding = 'SAME'))

	filter2 = tf.Variable(tf.random_normal(kernel_ , stddev = 0.05))
	layer1_conv_2 = tf.nn.relu(tf.nn.conv2d(layer1_conv_1 ,filter2 , strides = strides , padding = 'SAME'))

	layer1_maxpool = tf.nn.relu(tf.nn.max_pool(layer1_conv_2 , ksize = ksize , strides = [1,2,2,1] , padding = 'SAME'))
	print("Layer-1",layer1_maxpool.get_shape().as_list())

	#CONVOLUTION BLOCK - 2
	filter3 = tf.Variable(tf.random_normal(kernel2 , stddev = 0.05))
	layer2_conv_1 = tf.nn.relu(tf.nn.conv2d(layer1_maxpool ,filter3 ,strides = strides , padding = 'SAME'))

	filter4 = tf.Variable(tf.random_normal(kernel2_ , stddev = 0.05))
	layer2_conv_2 = tf.nn.relu(tf.nn.conv2d(layer2_conv_1 ,filter4 ,strides = strides , padding = 'SAME'))

	layer2_maxpool = tf.nn.relu(tf.nn.max_pool(layer2_conv_2 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer2 = tf.nn.dropout(layer2_maxpool , droupout_prob)
	print("Layer-2",layer2.get_shape().as_list())

	#CONVOLUTION BLOCK - 3
	filter5 = tf.Variable(tf.random_normal(kernel3 , stddev = 0.05))
	layer3_conv_1 = tf.nn.relu(tf.nn.conv2d(layer2 , filter5 ,strides = strides , padding ='SAME'))

	filter6 = tf.Variable(tf.random_normal(kernel3_ , stddev = 0.05))
	layer3_conv_2 = tf.nn.relu(tf.nn.conv2d(layer3_conv_1 , filter6 ,strides = strides , padding ='SAME'))

	filter7 = tf.Variable(tf.random_normal(kernel3_ , stddev = 0.05))
	layer3_conv_3 = tf.nn.relu(tf.nn.conv2d(layer3_conv_2 , filter7 ,strides = strides , padding ='SAME'))

	layer3_maxpool = tf.nn.relu(tf.nn.max_pool(layer3_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer3 = tf.nn.dropout(layer3_maxpool , droupout_prob)
	print("Layer-3",layer3.get_shape().as_list())
	shape = layer3.get_shape().as_list()

	#CONVOLUTION BLOCK - 4
	filter8 = tf.Variable(tf.random_normal(kernel4 , stddev = 0.05))
	layer4_conv_1 = tf.nn.relu(tf.nn.conv2d(layer3 , filter8 ,strides = strides , padding ='SAME'))

	filter9 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	layer4_conv_2 = tf.nn.relu(tf.nn.conv2d(layer4_conv_1 , filter9 ,strides = strides , padding ='SAME'))

	filter10 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	layer4_conv_3 = tf.nn.relu(tf.nn.conv2d(layer4_conv_2 , filter10 ,strides = strides , padding ='SAME'))

	layer4_maxpool = tf.nn.relu(tf.nn.max_pool(layer4_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer4 = tf.nn.dropout(layer4_maxpool , droupout_prob)
	print("Layer-4",layer4.get_shape().as_list())
	shape = layer4.get_shape().as_list()

	#CONVOLUTION BLOCK - 5
	filter11 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	layer5_conv_1 = tf.nn.relu(tf.nn.conv2d(layer4 , filter11 ,strides = strides , padding ='SAME'))

	filter12 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	layer5_conv_2 = tf.nn.relu(tf.nn.conv2d(layer5_conv_1 , filter12 ,strides = strides , padding ='SAME'))

	filter13 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	layer5_conv_3 = tf.nn.relu(tf.nn.conv2d(layer5_conv_2 , filter13 ,strides = strides , padding ='SAME'))

	layer5_maxpool = tf.nn.relu(tf.nn.max_pool(layer5_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer5 = tf.nn.dropout(layer5_maxpool , droupout_prob)
	print("Layer-5",layer5.get_shape().as_list())
	shape = layer5.get_shape().as_list()

	#FIRST DENSE LAYER
	#kernel4 = [shape[1] * shape[2] * shape[3],500]
	filter14 = tf.Variable(tf.random_normal(kernel6 , stddev = 0.05))
	shapestraight = [-1 , shape[1] * shape[2] * shape[3]]
	layer6_shaping = tf.reshape(layer5 , shapestraight)
	layer6drop = tf.nn.dropout(layer6_shaping , droupout_prob)
	layer6 = tf.nn.relu(tf.matmul(layer6drop , filter14))
	print("Layer-6",layer6.get_shape().as_list())

	#SECOND DENSE LAYER
	filter15 = tf.Variable(tf.random_normal(kernel7, stddev = 0.05))
	layer7 = tf.nn.relu(tf.matmul(layer6, filter15))
	print("Layer-7",layer7.get_shape().as_list())

	#OUTPUT LAYER
	filter16 = tf.Variable(tf.random_normal(kernel8 ,stddev = 0.05))
	#layer8drop = tf.nn.dropout(layer4 , droupout_prob)
	output = tf.matmul(layer7 , filter16)
	print("Output",output.get_shape().as_list())

	return output

kernel = [3,3,1,64]
kernel_ = [3,3,64,64]
kernel2 = [3,3,64,128]
kernel2_ = [3,3,128,128]
kernel3 = [3,3,128,256]
kernel3_ = [3,3,256,256]
kernel4 = [3,3,256,512]
kernel5 = [3,3,512,512]
kernel6 = [7 * 7 * 512,4096]
kernel7 = [4096,4096]
kernel8 = [4096 , 10]

droupout_prob = tf.placeholder(tf.float32)

train_layer = vgg(X, kernel, kernel_, kernel2, kernel2_, kernel3, kernel3_, kernel4, kernel5, kernel6, kernel7, kernel8, droupout_prob)

Y_ = tf.nn.softmax(train_layer)
correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = train_layer)
#optimize = tf.train.MomentumOptimizer(learning_rate = 1e-4, momentum = 0.9).minimize(cost)
optimize = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(init)

	# TRAINING PHASE
	for ep in range(40):
		print("Epoch number %d" %(ep))
		x = 0
		count = 0
		for i in range(int(len(tr_img_data)/batch_size)):
			testing = []
			trainimg = tr_img_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_lbl_data[batch_size * i : batch_size * (i+1)]
			for i in trainlbl:
				testing.append(np.reshape(i, (-1, 10)))
			testing = np.reshape(testing, (batch_size, 10))
			#print(testing.shape)
			#print(trainimg.shape)
			data = {X : trainimg , Y: testing , droupout_prob: 1.0}
			session.run(optimize , feed_dict = data)
			print(session.run(accuracy , feed_dict = data))

	# TESTING PHASE
	print("TESTING PHASE!!")
	testing = []
	for i in range(int(len(tst_img_data)/batch_size)):
		testing = []
		testimg = tst_img_data[batch_size * i : batch_size * (i+1)]
		testlbl = tst_lbl_data[batch_size * i : batch_size * (i+1)]
		for i in testlbl:
			testing.append(np.reshape(i, (-1, 10)))
		testing = np.reshape(testing, (batch_size, 10))
		#print(testing.shape)
		#print(trainimg.shape)
		data = {X : testimg , Y: testing , droupout_prob: 1.0}
		print(session.run(accuracy , feed_dict = data))
	'''for i in tst_lbl_data:
		testing.append(np.reshape(i, (-1, 10)))
	testing = np.reshape(testing, (len(tst_lbl_data), 10))
	data = {X : tst_img_data , Y : testing , droupout_prob : 1}'''

	print("after training : ")
	print(session.run(accuracy , feed_dict = data))




