from tensorflow.examples.tutorials.mnist import input_data
from scipy.io.arff import loadarff
import os, time, itertools, imageio, pickle, arff, random, csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, 9, kernel_initializer=w_init)
        o = tf.nn.tanh(dense2)

        return o

# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        lrelu1 = lrelu(dense1, 0.2)

        dense2 = tf.layers.dense(lrelu1, 1, kernel_initializer=w_init)
        o = tf.nn.sigmoid(dense2)

        return o, dense2


#===============================================================================
#training GAN

train_set = gan_train
train_label = gan_label

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 1000

# label preprocess
onehot = np.eye(2)

temp_z_ = np.random.normal(0, 1, (400, 100))
fixed_z_ = temp_z_
fixed_y_1 = np.zeros((200, 1))
fixed_y_2 = np.zeros((200, 1))
fixed_y_ = pd.concat([pd.DataFrame(fixed_y_1), pd.DataFrame(fixed_y_2)])
fixed_y_ = onehot[fixed_y_.astype(np.int32)].squeeze()

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 9))
y = tf.placeholder(tf.float32, shape=(None, 2))
z = tf.placeholder(tf.float32, shape=(None, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# results save folder
root = 'HW2_data_results/'
model = 'cGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(len(train_set) // batch_size):
        # update discriminator
        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
        y_ = train_label[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 100))
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, y: y_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        y_ = np.random.randint(0, 2, (batch_size, 1))
        y_ = onehot[y_.astype(np.int32)].squeeze()
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y: y_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    test_images = sess.run(G_z, {z: fixed_z_, y: fixed_y_, isTrain: False})
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")

np.savetxt("HW2_data_results/GAN-test_images-breast_cancer.csv", test_images, delimiter = ",")

sess.close()
