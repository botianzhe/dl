#%%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris=load_iris()
iris.keys()
X=iris['data'][:,2:]
y=iris['target']
model=Perceptron()
model.fit(X,y)
model.predict([[2,0.5]])

#%%
import tensorflow as tf
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',)

#%%
mnist

#%%
mnist.keys()

#%%
X=mnist['data']
y=mnist['target']
y=y.astype(np.int32)
y
#%%
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X)

#%%
feature_columns

#%%
from tensorflow.contrib.learn import DNNClassifier
model=DNNClassifier(hidden_units=[300,100],feature_columns=feature_columns,n_classes=10,)
model.fit(x=X,y=y,batch_size=50,steps=40000)

#%%
# 如果你在 MNIST 数据集上运行这个代码（在缩放它之后，例如，通过使用 skLearn
# 的 StandardScaler  ），你实际上可以得到一个在测试集上达到 98.1% 以上精度的模型！这比
# 我们在第 3 章中训练的最好的模型都要好：
from sklearn.metrics import accuracy_score
y_pred=list(model.predict(X))
print(accuracy_score(y,y_pred))

#%%
# TF.Learn 学习库也为评估模型提供了一些方便的功能
model.evaluate(X,y)

#%%
import tensorflow as tf

n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10
X=tf.placeholder(dtype=tf.float32,shape=(None,n_inputs),name='X')
y=tf.placeholder(dtype=tf.int64,shape=(None),name='y')
def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs=int(X.get_shape()[-1])
        stddev=2/np.sqrt(n_inputs)
        init=tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        W=tf.Variable(init,name='weights')
        b=tf.Variable(tf.zeros((n_neurons)),name='biases')
        z=tf.matmul(X,W)+b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z
    
with tf.name_scope('dnn'):
    hidden1_layer=neuron_layer(X,n_hidden1,'hidden1',activation='relu')
    hidden2_layer=neuron_layer(hidden1_layer,n_hidden2,'hidden2',activation='relu')
    logits=neuron_layer(hidden2_layer,n_outputs,'output')

# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
#     activation=tf.nn.relu)
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
#     activation=tf.nn.relu)
#     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

#%%



#%%
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
start= time.time()
mnist=input_data.read_data_sets('data/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
n_epochs=100
batch_size=50
n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10

X=tf.placeholder(dtype=tf.float32,shape=(None,n_inputs),name='X')
y=tf.placeholder(dtype=tf.int64,shape=(None),name='y')

def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs=int(X.get_shape()[-1])
        stddev=2/np.sqrt(n_inputs)
        init=tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        W=tf.Variable(init,name='weights')
        b=tf.Variable(tf.zeros((n_neurons)),name='biases')
        z=tf.matmul(X,W)+b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z
with tf.device('/gpu:0'):    
    with tf.name_scope('dnn'):
        hidden1_layer=neuron_layer(X,n_hidden1,'hidden1',activation='relu')
        hidden2_layer=neuron_layer(hidden1_layer,n_hidden2,'hidden2',activation='relu')
        logits=neuron_layer(hidden2_layer,n_outputs,'output')
    with tf.name_scope('loss'):
        xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss=tf.reduce_mean(xentropy,name='loss')

learning_rate=0.01
with tf.name_scope('train'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    train_op=optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init.run()
    for epoch in range(n_epochs):
        starta=time.time()
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch,y_batch=mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={X: X_batch,y:y_batch})
        acc_train=accuracy.eval(feed_dict={X: X_batch,y:y_batch})
        acc_test=accuracy.eval(feed_dict={X: mnist.test.images,y:mnist.test.labels})
        enda=time.time()
        print('Epoch ',epoch,'Train Accuracy:',acc_train,'Test Accuracy:',acc_test,'Time:',enda-starta)
    save_path=saver.save(sess,'model/model.ckpt')
    end=time.time()
    print("Total time:",end-start)


#%%
with tf.Session() as sess:
    saver.restore(sess,'model/model.ckpt')
    X_new=mnist.test.images
    Z=logits.eval(feed_dict={X:X_new})
    y_pred=np.argmax(Z,axis=1)
from sklearn.metrics import accuracy_score
print(accuracy_score(mnist.test.labels,y_pred))
#%%
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
from functools import partial
start= time.time()
mnist=input_data.read_data_sets('data/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
n_epochs=50
batch_size=50
n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10
batch_norm_momentum = 0.9
X=tf.placeholder(dtype=tf.float32,shape=(None,n_inputs),name='X')
y=tf.placeholder(dtype=tf.int64,shape=(None),name='y')
training=tf.placeholder_with_default(False,shape=(),name='training')

with tf.name_scope('dnn'):
    he_init=tf.contrib.layers.variance_scaling_initializer()
    my_batch_norm_layer = partial(
        tf.layers.batch_normalization,
        training = training,
        momentum = batch_norm_momentum
    )
    my_dense_layer = partial(
        tf.layers.dense,
        kernel_initializer = he_init
    )
    hidden1_layer=my_dense_layer(X,n_hidden1,name='hidden1')
    bn1=tf.nn.elu(my_batch_norm_layer(hidden1_layer))
    hidden2_layer=my_dense_layer(bn1,n_hidden2,name='hidden2')
    bn2=tf.nn.elu(my_batch_norm_layer(hidden2_layer))
    logits_before_bn=my_dense_layer(bn2,n_outputs,name='output')
    # logits=my_batch_norm_layer(logits_before_bn)
    logits=logits_before_bn

with tf.name_scope('loss'):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name='loss')

learning_rate=0.01
with tf.name_scope('train'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    train_op=optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
init=tf.global_variables_initializer()
saver=tf.train.Saver()
config=tf.ConfigProto()
config.log_device_placement=True
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        starta=time.time()
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch,y_batch=mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={X: X_batch,y:y_batch})
        acc_train=accuracy.eval(feed_dict={training:True,X: X_batch,y:y_batch})
        acc_test=accuracy.eval(feed_dict={X: mnist.test.images,y:mnist.test.labels})
        enda=time.time()
        print('Epoch ',epoch,'Train Accuracy:',acc_train,'Test Accuracy:',acc_test,'Time:',enda-starta)
    save_path=saver.save(sess,'model/model.ckpt')
    end=time.time()
    print("Total time:",end-start)
#%%
threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)
#%%
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
scope="hidden[123]") # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict) # to restore layers 1-3
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs): # not shown in the book
        for iteration in range(mnist.train.num_examples // batch_size): # not shown
            X_batch, y_batch = mnist.train.next_batch(batch_size) # not shown
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) # not shown
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels}) # not shown
     print(epoch, "Test accuracy:", accuracy_val) # not shown
    save_path = saver.save(sess, "./my_new_model_final.ckpt")

#%%
def variables_on_gpu(op):
    if op.type=='Variable':
        return '/gpu:0'
    else:
        return '/cpu:0'

with tf.device(variables_on_gpu):
    a=tf.Variable(2)
    b=tf.constant(2)
    