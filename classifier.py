'''	Important Note for running/training the model
In order to actually train the model:
-line call to os.chdir() must be able to navigate to the correct data directory, can be set where ever as long as properly found
-the data must be stored in the following manner "./record/train/*","./gt/train/*","./record/test/*","./gt/test/*". So in this case, '/practice_data/' has two folders, /practice_data/record and /practice_data/gt, for recording and ground truth. Then each of these has a /train & /test folder. This could be modified by appropriately changing data_loader.py to fit desired action
-heith and bin_size must be set correctly the the height and width of the "image", the numpy array spectrogram chunks, that will be fed in to train, otherwise the network will not run.
	-remember that each gt file for a corresponding data chunk must be a 3 element array, ie [ 1, 0, 0 ] to represent whether the bin has nothing, a car or people walking. This could be changed to 2 elements, car or people, by feeding no bins with nothing, however since the intent was to do real time processing it was designed with the idea that it would have the option to choose nothing present
-Solved I believe, although log folder must exist- The log path must be appropriately set to log information needed for tensorboard
	-# *** Also note that logs must be started fresh each iteration, otherwise tensorboard will regraph old logs, so delete logs in between runs ***
-Solved I believe, although the folder model must exist- The model_path must be appropriately set in order to be able to reload the model to run later, ie realtime processing
-also num_iterations should be set to something much higher. I do not know what exactly, but I think maybe 20000? idk this could be a trial and error thing
'''


import tensorflow as tf
import sys, os
sys.path.append('./')
from data_loader import *

path = os.getcwd()
data = DataLoader()

os.chdir('../data/network_data/') # go to directory where data is located
#path = os.getcwd()

data.loadData()

train_path = ''	# path to training data
test_path = ''	# path to test data

# how many times to run the model
num_iterations = 100

# height and width of intensity maps passed in (divisible by 16, 2^(# of max_pools))
height = 794
bin_size = 48

# data height and width after encoding conv layers
end_h = int(height/4)
end_bin_size = int(bin_size/4)

with tf.name_scope("input"):
# x holds intensity maps, shape=(# of examples, TBD, TBD, 1 channel)
	x = tf.placeholder(tf.float32, shape=(None, height, bin_size,1), name="input")

# holds the ground truth of the network about what the example is in one hot vector
# outputs can be 1 of 3 things= nothing, human walking, or car. ie [1, 0, 0] = nothing
with tf.name_scope("labeled_data"):
	y_ = tf.placeholder(tf.float32, [None, 2], name="labeled")

keep_prob = tf.placeholder(tf.float32)

def weights(shape,layer):
	# variable filled with random normally distributed number with shape of "shape"
	with tf.name_scope("weights"):	
		initial = tf.random_normal(shape)
		weights_name = "weights"+layer
		return tf.Variable(initial, name=weights_name)

def biases(shape,layer):
	with tf.name_scope("bias"):
		initial = tf.random_normal(shape)
		bias_name = "bias"+layer
		return tf.Variable(initial, name=bias_name)

def convolution(x_, W, b, s=1):
	x_ = tf.nn.conv2d(x_, W, strides=[1,s,s,1], padding='SAME')
	x_ = tf.nn.bias_add(x_, b)
	return tf.nn.relu(x_)

def maxpool(x_, k=2):
	return tf.nn.max_pool(x_, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def network(x, weights, biases, dropout):
	# 3x3 conv size, 1 input channel, 32 output feature channels
	c1_W = weights([3,3,1,32],"_1")
	# biases for each feature channel
	c1_b = biases([32],"_1")
	# convolve input with the weights and add biases
	c1_conv = convolution(x, c1_W, c1_b)
	# pool the maxes from 2x2 windows of the convolution output
	c1_pool = maxpool(c1_conv)

	# 3x3 conv size, 32 input channels, 64 output feature channels
	c2_W = weights([3,3,32,64],"_2")
	c2_b = biases([64],"_2")
	c2_conv = convolution(c1_pool, c2_W, c2_b)
	c2_pool = maxpool(c2_conv)

	c3_W = weights([3,3,64,128],"_3")
	c3_b = biases([128],"_3")
	c3_conv = convolution(c2_pool, c3_W, c3_b)
	c3_pool = maxpool(c3_conv)

	c4_W = weights([3,3,128,256],"_4")
	c4_b = biases([256],"_4")
	c4_conv = convolution(c3_pool, c4_W, c4_b)
	c4_pool = maxpool(c4_conv)
	# end of convolution layers

	# beginning of fully connected layers
	# image size is now reduced to 1/4 the orginal size with 256 features
	# fully connected layer with 1024 neurons
	fc1_W = weights([end_h*end_bin_size*256, 1024],"_FC1")	
	fc1_b = biases([1024],"_FC1")

	# flatten out final pooled layer before connecting to fully connected layer
	c4_pool = tf.reshape(c4_pool, [-1, end_h*end_bin_size*256])
	fc1_out = tf.nn.relu(tf.matmul(c4_pool, fc1_W))
	fc1_out = tf.nn.bias_add(fc1_out, fc1_b)

	# will be defined while training

	# dropout to prevent overfitting
	fc1_out_dropout = tf.nn.dropout(fc1_out, keep_prob)

	# second fully connected layer
	fc2_W = weights([1024, 2],"_FC2")
	fc2_b = biases([2],"_FC2")

	y_out = tf.matmul(fc1_out_dropout, fc2_W)
	with tf.name_scope("output"):
		y_out = tf.nn.bias_add(y_out, fc2_b)
		# commented out because of infinity in histogram error, probably / by 0
		#tf.summary.histogram("output", y_out)
	
	# return the predictions	
	return y_out

prediction = network(x, weights, biases, .05)


# cost, compute softmax cross entropy between the prediction and GT
with tf.name_scope("cross_entropy"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_), name="cross")
	tf.summary.scalar("cross_entropy", cross_entropy)

# minimze the cross entropy with gradient descent with step of 0.5
with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# tf.argmax() returns index of max
# tf.equal() returns element-wise x==y bools
correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
# computes the percentage of predictions that were correct
with tf.name_scope("accuracy"):
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
	tf.summary.scalar("accuracy", accuracy)

# create a session to run model
sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
merged = tf.summary.merge_all()

# set log_path to desired path for logs
# *** Also note that logs must be started fresh each iteration, otherwise tensorboard
# will regraph old logs, so delete logs in between files ***
log_path = "../../net_classifier/logs/"
writer = tf.summary.FileWriter(log_path, graph=sess.graph)

sess.run(init)


# not currently included because getBatch() is not yet defined
for i in range(num_iterations):
	batch = data.getBatch(10)
	sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})	
	if i % 1 == 0:
		train_accuracy, result = sess.run([accuracy, merged], feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})		
		# train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
		print("%d: Accuracy= %g" % (i, train_accuracy))
		#result = sess.run(merged, feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})
		writer.add_summary(result, i)
	#train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})


print("Training Concluded")

testBatch = data.getTest(0,all=True)

print("test accuracy %g" % sess.run(accuracy, feed_dict={x:testBatch[0], y_:testBatch[1], keep_prob: 0.5}))

# set model_path to where you want to save the model
model_path = "../../net_classifier/model/sstdc_classifier.ckpt"
save_path = saver.save(sess, model_path)


