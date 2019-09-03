import tensorflow as tf 
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt

china=load_sample_image("china.jpg")
flower=load_sample_image('flower.jpg')
dataset=np.array([china,flower],dtype=np.float32)
batch_size,height,width,channels=dataset.shape
print(dataset.shape)
filters=np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters[:,3,:,0]=1
filters[3,:,:,1]=1
X=tf.placeholder(shape=(None,height,width,channels),dtype=tf.float32)
cnn=tf.nn.conv2d(X,filters,[1,2,2,1],padding='SAME')
pool=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
with tf.Session() as sess:
    output=sess.run(pool,feed_dict={X:dataset})
print(output.shape)
plt.imshow(output[1,:,:,0],cmap='gray')
plt.show()

#%%
