#%%
# 创建第一个图
import tensorflow as tf
a=tf.Variable(3,name='a')
b=tf.Variable(2,name='b')
f=a*b+2

#%%
# way1
sess=tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
print(sess.run(f))
sess.close()
#%%
# way2
# 在 with  块中，会话被设置为默认会话。 调用 x.initializer.run()  等效于调
# 用 tf.get_default_session().run(x.initial)  ， f.eval()  等效于调
# 用 tf.get_default_session().run(f)  。 这使得代码更容易阅读。 此外，会话在块的末尾自动
# 关闭。
with tf.Session() as sess:
    a.initializer.run()
    b.initializer.run()
    res=f.eval()
print(res)

#%%
# way3
init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(f.eval())


#%%
# way4
init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
init.run()
print(f.eval())

#%%
a.graph is tf.get_default_graph()

#%%
graph=tf.Graph()
with graph.as_default():
    x=tf.Variable(3)
print(x.graph is graph)
print(x.graph is tf.get_default_graph())

#%%
import numpy as np
from sklearn.datasets import fetch_california_housing

housing=fetch_california_housing()
m,n=housing.data.shape
housing_plus_bias=np.c_[np.ones((m,1)),housing.data]
print(housing_plus_bias.shape)
X=tf.constant(housing_plus_bias,name='x',dtype=tf.float32)
y=tf.constant(housing.target.reshape(-1,1),name='y',dtype=tf.float32)
XT=tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    print(theta.eval())
#%%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
housing=fetch_california_housing()
m,n=housing.data.shape
housing_plus_bias=np.c_[np.ones((m,1)),housing.data]
scale=StandardScaler()
housing_data=scale.fit_transform(housing_plus_bias)
epochs=1000
learning_rate=0.1
X=tf.constant(housing_data,name='x',dtype=tf.float32)
y=tf.constant(housing.target.reshape(-1,1),name='y',dtype=tf.float32)
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
y_pred=tf.matmul(X,theta,name='predictions')
error=y_pred - y
mse=tf.reduce_mean(tf.square(error),name='mse')
# grandient=1/m*tf.matmul(tf.transpose(X),error)
# train_op=tf.assign(theta,theta-grandient*learning_rate)
# grandient = tf.gradients(mse, [theta])[0]
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(mse)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        if epoch%100==0:
            print('Epoch:',epoch,'MSE:',mse.eval())
        sess.run(train_op)
    best_theta=sess.run(theta)
    print(best_theta)


#%%
A=tf.placeholder(dtype=tf.float32,name='A')
B=A+2
with tf.Session() as sess:
    B_1=B.eval(feed_dict={A:[[1,2,3]]})
    B_2=B.eval(feed_dict={A:[[1,2,3],[2,3,4]]})
print(B_1,B_2)

#%%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


housing=fetch_california_housing()
m,n=housing.data.shape

scale=StandardScaler()
housing_data=scale.fit_transform(housing.data)
housing_plus_bias=np.c_[np.ones((m,1)),housing_data]
batch_size=100
n_batches=int(np.ceil(m/batch_size))
epochs=10
learning_rate=0.01




X=tf.placeholder(dtype=tf.float32,shape=(None,n+1),name='X')
y=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y')
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42),name='theta')

y_pred=tf.matmul(X,theta,name='predictions')
error=y_pred - y
mse=tf.reduce_mean(tf.square(error),name='mse')
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(mse)
init=tf.global_variables_initializer()
# 定义批量大小并计算总批次数


def fetch_batch(epoch,batch_index,batch_size):
    # print(epoch,batch_index,batch_size)
    know=np.random.seed(epoch*n_batches+batch_index)
    indices=np.random.randint(m,size=batch_size)
    X_batch=housing_plus_bias[indices]
    y_batch=housing.target.reshape(-1,1)[indices]
    # print(X_batch.shape)
    return X_batch,y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        for batch_index in range(n_batches):
            X_batch,y_batch=fetch_batch(epoch,batch_index,batch_size)
            _,mse_value=sess.run([train_op,mse],feed_dict={X:X_batch,y:y_batch})
        print('Epoch:',epoch,'MSE:',mse_value)
    best_theta=theta.eval()
    print(best_theta)



#%%
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m,n))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size)) # ceil() 方法返回 x 的值上限 - 不小于 x 的最小整数
def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index) # not shown in the book
    # print("我是know:",know)
    indices = np.random.randint(m, size=batch_size) # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            _,mse_value=sess.run([training_op,mse],feed_dict={X:X_batch,y:y_batch})
        print('Epoch:',epoch,'MSE:',mse_value)
    save_path=saver.save(sess,'model/model.ckpt')
        
    best_theta = theta.eval()
        
print(best_theta)

#%%
from datetime import datetime
now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir='./log'
log_dir='{}/root-{}/'.format(root_logdir,now)
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m,n))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
mse_summary=tf.summary.scalar('mse',mse)
file_writer=tf.summary.FileWriter(log_dir,tf.get_default_graph())  
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size)) # ceil() 方法返回 x 的值上限 - 不小于 x 的最小整数
def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index) # not shown in the book
    # print("我是know:",know)
    indices = np.random.randint(m, size=batch_size) # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            summary_str=mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
            step = epoch * n_batches + batch_index
            file_writer.add_summary(summary_str,step)
            _,mse_value=sess.run([training_op,mse],feed_dict={X:X_batch,y:y_batch})
        print('Epoch:',epoch,'MSE:',mse_value)
    save_path=saver.save(sess,'model/model.ckpt')
        
    best_theta = theta.eval()
file_writer.close()    
print(best_theta)

#%%
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m,n))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = r"D://tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index) # not shown in the book
    indices = np.random.randint(m, size=batch_size) # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess: # not shown in the book
    sess.run(init) # not shown
    for epoch in range(n_epochs): # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
file_writer.close()
print(best_theta)

#%%
