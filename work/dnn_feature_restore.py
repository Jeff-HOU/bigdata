import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import scale
import time
import os
from datetime import date
var_save_dir = './saved_vars'
tfeature_save_file = var_save_dir + '/tfeature.npy'
sfeature_save_file = var_save_dir + '/sfeature.npy'
tlabel_save_file = var_save_dir + '/tlabel.npy'
log_dir = './log'
tfeature = scale(np.load(tfeature_save_file))
sfeature = scale(np.load(sfeature_save_file))
tlabel = scale(np.load(tlabel_save_file))
tlabel_squeeze = np.squeeze(tlabel, axis=1).astype(int)
labels = np.asarray(pd.get_dummies(tlabel_squeeze))

train_val_split = np.random.rand(len(tfeature)) < 0.70
train_x = tfeature[train_val_split]
train_y = labels[train_val_split]
val_x = tfeature[~train_val_split]
val_y = labels[~train_val_split]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

n_input = 16
n_hidden_1 = 8
#n_hidden_2 = 75
n_hidden_3 = 4

n_classes = labels.shape[1]

learning_rate = 0.003
training_epochs = 20
batch_size = 10

total_batches = tfeature.shape[0] # batch_size

X = tf.placeholder(tf.float32, shape=[None,n_input])
Y = tf.placeholder(tf.float32,[None,n_classes])

# --------------------- Encoder Variables --------------- #

e_weights_h1 = weight_variable([n_input, n_hidden_1])
e_biases_h1 = bias_variable([n_hidden_1])

#e_weights_h2 = weight_variable([n_hidden_1, n_hidden_2])
#e_biases_h2 = bias_variable([n_hidden_2])

e_weights_h3 = weight_variable([n_hidden_1, n_hidden_3])
e_biases_h3 = bias_variable([n_hidden_3])

# --------------------- Decoder Variables --------------- #

d_weights_h1 = weight_variable([n_hidden_3, n_hidden_1])
d_biases_h1 = bias_variable([n_hidden_1])

#d_weights_h2 = weight_variable([n_hidden_2, n_hidden_1])
#d_biases_h2 = bias_variable([n_hidden_1])

d_weights_h3 = weight_variable([n_hidden_1, n_input])
d_biases_h3 = bias_variable([n_input])

# --------------------- DNN Variables ------------------ #

dnn_weights_h1 = weight_variable([n_hidden_3, n_hidden_1])
dnn_biases_h1 = bias_variable([n_hidden_1])

#dnn_weights_h2 = weight_variable([n_hidden_2, n_hidden_1])
#dnn_biases_h2 = bias_variable([n_hidden_1])

dnn_weights_out = weight_variable([n_hidden_1, n_classes])
dnn_biases_out = bias_variable([n_classes])

def encode(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,e_weights_h1),e_biases_h1))
    #l2 = tf.nn.tanh(tf.add(tf.matmul(l1,e_weights_h2),e_biases_h2))
    l3 = tf.nn.tanh(tf.add(tf.matmul(l1,e_weights_h3),e_biases_h3))
    return l3
    
def decode(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,d_weights_h1),d_biases_h1))
    #l2 = tf.nn.tanh(tf.add(tf.matmul(l1,d_weights_h2),d_biases_h2))
    l3 = tf.nn.tanh(tf.add(tf.matmul(l1,d_weights_h3),d_biases_h3))
    return l3

def dnn(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,dnn_weights_h1),dnn_biases_h1))
    #l2 = tf.nn.tanh(tf.add(tf.matmul(l1,dnn_weights_h2),dnn_biases_h2))
    out = tf.nn.softmax(tf.add(tf.matmul(l1,dnn_weights_out),dnn_biases_out))
    return out

encoded = encode(X)
decoded = decode(encoded) 
y_ = dnn(encoded)

us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
s_cost_function = -tf.reduce_sum(Y * tf.log(y_))

reg_constant = 0.1
regularizer = tf.nn.l2_loss(dnn_weights_out) + tf.nn.l2_loss(dnn_weights_h1)
s_cost_function = tf.reduce_mean(s_cost_function + regularizer * reg_constant)

us_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(us_cost_function)
s_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(s_cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, "./log/20170625-130102.ckpt")
    print("Model restored.")
    
    prediction=tf.argmax(y_,1)
    predict = prediction.eval(feed_dict={X: sfeature}, session=session)
    np.set_printoptions(threshold=np.nan)
    print ("predictions\n", predict)
    num_of_black_samples = 100000 - np.count_nonzero(predict)
    black_samples = np.zeros(num_of_black_samples)
    print("black_samples: " + str(num_of_black_samples))
    j = 0
    k = 1
    for i in range(100000):
        if predict[i] == 0:
            black_samples[j] = k
            j += 1
        k += 1
    d = date.today().timetuple()
    fname = '../submit/BDC0539_' + str(d[0]).zfill(4) + str(d[1]).zfill(2) + str(d[2]).zfill(2) + '.txt'
    np.savetxt(fname, black_samples, fmt='%d', delimiter='\n')
    #print ("probabilities\n", y_.eval(feed_dict={X: sfeature}, session=session))
    #print ("\nTesting Accuracy:", session.run(accuracy, feed_dict={X: sfeature, Y: test_labels}))
