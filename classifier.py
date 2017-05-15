import tensorflow as tf

train_path = ''	# path to training data
test_path = ''	# path to test data

# height and width of intensity maps passed in (divisible by 16, 2^(# of max_pools))
height = 128
bin_size = 128

# data height and width after encoding conv layers
end_h = int(height/16)
end_bin_size = int(bin_size/16)

# x holds intensity maps, shape=(# of examples, TBD, TBD, 1 channel)
x = tf.placeholder(tf.float32, shape=(None, height, width, 1))

# holds the ground truth of the network about what the example is in one hot vector
# outputs can be 1 of 3 things= nothing, human walking, or car. ie [1, 0, 0] = nothing
y_ = tf.placeholder(tf.float32, [None, 3])

def weights(shape):
	# variable filled with random normally distributed number with shape of "shape"
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def biases(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def convolution(x_, W, b, s=1):
	x_ = tf.nn.conv2d(x_, W, strides=[1,s,s,1], padding='SAME')
	x_ = tf.nn.bias_add(x_, b)
	return tf.nn.relu(x_)

def maxpool(x_, k=2):
	return tf.nn.max_pool(x_, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def network(x, weights, biases, dropout):
	# 3x3 conv size, 1 input channel, 32 output feature channels
	c1_W = weights([3,3,1,32])
	# biases for each feature channel
	c1_b = biases([32])
	# convolve input with the weights and add biases
	c1_conv = convolution(x, c1_W, c1_b)
	# pool the maxes from 2x2 windows of the convolution output
	c1_pool = maxpool(c1_conv)

	# 3x3 conv size, 32 input channels, 64 output feature channels
	c2_W = weights([3,3,32,64])
	c2_b = biases([64])
	c2_conv = convolution(c1_pool, c2_W, c2_b)
	c2_pool = maxpool(c2_conv)

	c3_W = weights([3,3,64,128])
	c3_b = biases([128])
	c3_conv = convolution(c2_pool, c3_W, c3_b)
	c3_pool = maxpool(c3_conv)

	c4_W = weights([3,3,128,256])
	c4_b = biases([256])
	c4_conv = convolution(c3_pool, c4_W, c4_b)
	c4_pool = maxpool(c4_conv)
	# end of convolution layers

	# beginning of fully connected layers
	# image size is now reduced to 1/4 the orginal size with 256 features
	# fully connected layer with 1024 neurons
	fc1_W = weights([end_h*end_bin_size*256, 1024])	
	fc1_b = biases([1024])

	# flatten out final pooled layer before connecting to fully connected layer
	c4_pool = tf.reshape(c4_pool, [-1, end_h*end_w*256])
	fc1_out = tf.nn.relu(tf.matmul(c4_pool, fc1_W))
	fc1_out = tf.nn.bias_add(fc1_out, fc1_b)

	# will be defined while training
	keep_probability = tf.placeholder(tf.float32)

	# dropout to prevent overfitting
	fc1_out_dropout = tf.nn.dropout(fc1_out, keep_probability)

	# second fully connected layer
	fc2_W = weights([1024, 3])
	fc2_b = biases([3])

	y_out = tf.matmul(fc1_out_dropout, fc2_W)
	y_out = tf.nn.bias_add(y_out, fc2_b)
	
	# return the predictions	
	return y_out

prediction = network(x, weights, biases, .05)

# cost, compute softmax cross entropy between the prediction and GT
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_))
# minimze the cross entropy with gradient descent with step of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# tf.argmax() returns index of max
# tf.equal() returns element-wise x==y bools
correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
# computes the percentage of predictions that were correct
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# create a session to run model
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

'''
for i in range(10000):
	batch = getBatch(10)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
		print("%d: Accuracy= %g" % (i, train_accuracy))
	train_step.run(feed_dict={x:batch[0], y_:batch[1]})
'''


print("hello")
