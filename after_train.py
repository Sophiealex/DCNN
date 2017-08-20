# -*- coding: utf-8 -*-

import tensorflow as tf
from model import *
import dataUtils
import numpy as np
import time
import os

# start of print time cost
import datetime
starttime = datetime.datetime.now()
#long running
#do something other


embed_dim = 48 # wordembedding's dim
kernel_size = [3, 3] # kernel size
top_k = 4 # wait to change
k1 = 30 # sent_length / 2
num_filters = [12, 8] # each conv_layer's num_filters
val = 300 # validation set
batch_size= 50
n_epochs= 30
embed_learn_batches = 1080 # fix embedding_layer after 1080 batches len(vocabulary) / batch_size * 10
num_hidden= 100 # 想要得到的隐含向量维度 作者得到的是对480维，这里我设置为100
sentence_length= 59 # 62 去掉标点前为62，去掉标点后为59
lr = 0.001
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 3
low_dim = 30 # binary_y's dim
k_top = 5
weight_decay = 5e-5 # weight_decay on regularization item

# import data
print("loading data...")
x_, vocabulary, _inv, test_size = dataUtils.load_data()
y_ = np.loadtxt("data/Binary_Z.txt")
# x_ :长度为句子个数的np.array，句子长度为sentence_length，不足的用padding补齐
# y_ :长度为句子个数的np.array，需要拟合的binary向量，从matlab里面得到

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(tf.cast(shape, tf.int32), stddev=0.01, mean=0), name=name)

sent = tf.placeholder(tf.int32, [None, sentence_length], name='sent')
binary_y = tf.placeholder(tf.int32, [None, low_dim], name='binary_code')
dropout_keep_prob = tf.placeholder(tf.float32, name='dropout')

with tf.name_scope("embedding_layer"):
    # W is word_embedding we need to use word2vec to initialize it
    # r = tf.sqrt(tf.cast(6 / embed_dim, dtype=tf.float32))
    # Get W_embeding
    W_embedding_input = dataUtils.get_wordembedding(vocabulary)
    # W_embedding_input = tf.convert_to_tensor(W_embedding_input, name='embedding_input')
    # W_embedding = tf.placeholder(tf.float32, [len(vocabulary), embed_dim])
    W_embedding = tf.Variable(W_embedding_input, name='embedding_layer')
    # tf.nn.embedding_lookup(W, sent) return a Tensor with the same type as the tensors W
    sent_embed = tf.nn.embedding_lookup(W_embedding, sent)
    input_x = tf.expand_dims(sent_embed, -1)
    # [batch_size, sentence_length, embed_dim, 1]

W1 = init_weights([kernel_size[0], embed_dim, 1, num_filters[0]], name='W1')
b1 = tf.Variable(tf.zeros(shape=[num_filters[0], embed_dim]), name='b1')

W2 = init_weights([kernel_size[1], embed_dim/2, num_filters[0], num_filters[1]], name='W2')
b2 = tf.Variable(tf.zeros(shape=[num_filters[1], embed_dim]), name='b2')

Wh = init_weights([k_top*num_filters[1]*embed_dim/4, num_hidden], name='Wh')
bh = tf.Variable(tf.zeros(shape=[num_hidden]), name='bh')

Wo = init_weights([num_hidden, low_dim], name='Wo')

model = DCNN(batch_size, sentence_length, num_filters, embed_dim, k_top, k1)
out = model.DCNN_2layer(input_x, W1, W2, b1, b2, k1, k_top, Wh, bh, Wo, dropout_keep_prob)

with tf.name_scope('cost'):
    pro_out = tf.sigmoid(out)
    cost = tf.losses.log_loss(predictions=pro_out, labels=binary_y) / batch_size # tf.cast(binary_y.shape[0], tf.float32)
    cost += 0.5 * weight_decay * (tf.reduce_mean(tf.square(Wh)) + tf.reduce_mean(tf.square(Wo))
                                  + tf.reduce_mean(tf.square(W1)) + tf.reduce_mean(tf.square(W2)))

saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints) # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads_and_vars = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "1503115104"))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    print(checkpoint_prefix)
    saver.restore(sess, checkpoint_prefix + "-" + str(1100))
    feed_dict = {
        sent: x_,
        binary_y: y_,
        dropout_keep_prob: 1.0
    }
    hidden_features = sess.run(model.get_hidden_feature(input_x, W1, W2, b1, b2, k1, k_top, Wh, bh), feed_dict=feed_dict)
    np.savetxt("hidden_features.txt", hidden_features)

    # end of print time cost
    endtime = datetime.datetime.now()
    print("Total cost {} seconds.".format((endtime - starttime).seconds))

    # Then go to matlab,
    # read the data,
    # use k-means to get labels file,
    # downsampling the label data,
    # get the t-sne figure and the output sentnece!
