import tensorflow as tf

# train_path = # path to training data
# test_path = # path to test data


# height and width of the intensity maps being passed in (make divisible by 4)
height = 100
width = 100
# x holds intensity maps, shape=(# of examples, TBD, TBD, 1 channel)
x = tf.placeholder(tf.float32, shape=(None, height, width, 1))

# holds the ground truth of the network about what the example is in one hot vector
# outputs can be 1 of 3 things= nothing, human walking, or car. ie [1, 0, 0] = nothing
y_ = tf.placeholder(tf.float32, [None, 3])

def weights(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def biases(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def convolution(x, W, b, s=1):
	x = tf.nn.conv2d(x, W, strides=[1,k,k,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], stride=[1,k,k,1], padding='SAME')

def network(x, weights, biases, dropout):
	c1_W = weights([3,3,1,32])
	c1_b = biases([32])
	c1_conv = convolution(x, c1_W, c1_b)
	c1_pool = maxpool(c1_conv)

	c2_W = weights([3,3,32,64])
	c2_b = biases([64])
	c2_conv = convolution(c1_pool, c2_W, c2_b)
	c2_pool = maxpool(c1_conv)

	c3_W = weights([3,3,64,128])
	c3_b = biases([128])
	c3_conv = convolution(c2_pool, c3_W, c3_b)
	c3_pool = maxpool(c3_conv)

	c4_W = weights([3,3,128,256])
	c4_b = biases([256])
	c4_conv = convolution(c3_pool, c4_W, c4_b)
	c4_pool = maxpool(c4_conv)

	'''
	# image size is now reduced to 1/4 the orginal size with 256 features
	# fully connected layer with 1024 neurons
	fc1_W = weights([(height/4)*(width/4)*256, 1024])	
	fc1_b = biases([1024])

	fc1_output = tf.nn.relu(tf.matmul(c4_pool, fc1_W))
	fc1_output = tf.nn.bias_add(fc1_ouptut, fc1_b)

	# will be defined while training
	keep_probability = tf.placeholder(tf.float32)
	fc1_output_dropout = tf.nn.dropout(fc1_output, keep_probability)

	# second fully connected layer
	'''







print("hello")
