#encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
##Defining weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
##Defining Bias
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
##Defining Convolution
def conv2d(x,W):
    #srtide[1,x_movement,y_,movement,1]
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding='SAME')
    #x: input，W:convolution parameter，[5,5,1,32] means 5*5 convolution core，1 channel，32 cores.strides:Moving step size of model，SAME & VALID: two kinds of padding，valid: extract from original images，same: extract from the original images with filling zeros，has same size with the original image.
##Defining Pooling
def max_pool_2x2(x):
    #ksize   strides
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
##Compute accuracy
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result
##Input
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])#10 output
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])
#-1 means the uncertainty of sample size，image size: 28*28
#print(x_image.shape)
##Convolution layer conv1
W_conv1 = weight_variable([5,5,1,32])#first convolution layer：core siez :5x5,1 color channel，32 convolution cores
b_conv1 = bias_variable([32])#bias for the first layer
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output of first layer：28x28x32
h_pool1 = max_pool_2x2(h_conv1)#output: 14x14x32
##Convolution layer conv2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)#output: 7x7x64
##The number of nodes in full connection layer and hidden layer is 1024.
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#make the image feature into 1 dimension data[n_samples,7,7,64]->>[n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)#Nonlinear activation function
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)#preventing overfitting
##softmaxe layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#loss,cross_entropy+softmax can produce the calssification algorithm
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
#initialization
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)#get the 100 data from where you download，mini_batch
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
