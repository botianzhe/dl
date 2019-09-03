#%%
import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    n_steps=20
    n_inputs = 1
    n_neurons = 100
    n_outputs=1

    # X0 = tf.placeholder(tf.float32, [None, n_inputs])
    # X1 = tf.placeholder(tf.float32, [None, n_inputs])
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    # Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
    # Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
    # b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
    # Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    # Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
    basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)
    # output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],dtype=tf.float32)
    output_seqs, states = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)
    # Y0,Y1=output_seqs
    init = tf.global_variables_initializer()
    # Mini-batch: instance 0,instance 1,instance 2,instance 3
    # X_batch = np.array([
    #     [[0, 1, 2], [9, 8, 7]], # instance 1
    #     [[3, 4, 5], [0, 0, 0]], # instance 2
    #     [[6, 7, 8], [6, 5, 4]], # instance 3
    #     [[9, 0, 1], [3, 2, 1]], # instance 4
    # ])
    sequence = [0.] * n_steps
    with tf.Session() as sess:
        init.run()
        for iteration in range(300):
            X_batch=np.array(sequence[-n_steps:]).reshape(1,n_steps,1)
            output= sess.run(output_seqs, feed_dict={X:X_batch})
            sequence.append(output[0,-1,0])
    print(output,'\n')
    # print(Y1_val)
    
#%%
import numpy as np
import tensorflow as tf
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.001
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
#%%
bcell=tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
bcell_drop=tf.nn.rnn_cell.DropoutWrapper(bcell,input_keep_prob=0.5)
mcell=tf.nn.rnn_cell.MultiRNNCell([bcell_drop]*10)
outputs,state=tf.nn.dynamic_rnn(mcell,X,dtype=tf.float32)
