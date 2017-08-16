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
# vocabulary :长度为5397的字典，说明语料库中一共包含5397个单词。key是单词，value是索引(去掉了标点符号)
# test_size :500，测试集大小

# shuffle数据
x, x_test = x_[:-test_size], x_[-test_size:]
y, y_test = y_[:-test_size], y_[-test_size:]
shuffle_index = np.random.permutation(np.arange(len(x)))
x_shuffled = x[shuffle_index]
y_shuffled = y[shuffle_index]

x_train, x_val = x_shuffled[:-val], x_shuffled[-val:]
y_train, y_val = y_shuffled[:-val], y_shuffled[-val:]

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

print('Start training')
with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads_and_vars = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss
    loss_summary = tf.summary.scalar("loss", cost)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Val summaries
    val_summary_op = tf.summary.merge([loss_summary])
    val_summary_dir = os.path.join(out_dir, "summaries", "val")
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

    # Checkpoint directory.
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # train model
    def train_step(x_batch, y_batch, total_step):
        feed_dict = {
            sent: x_batch,
            binary_y: y_batch,
            dropout_keep_prob: 0.5
        }
        _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, cost], feed_dict=feed_dict)
        print("TRAIN step {}/{}, loss {:g}".format(step,total_step, loss))
        train_summary_writer.add_summary(summaries, step)
        pass

    # Evaluates model on a val set
    def val_step(x_batch, y_batch, writer=None):
        feed_dict = {
            sent: x_batch,
            binary_y : y_batch,
            dropout_keep_prob: 1.0
        }
        step, summaries, loss = sess.run(
            [global_step, val_summary_op, cost],
            feed_dict)
        print("VALID step {}, loss {:g}".format(step, loss))
        if writer:
            writer.add_summary(summaries, step)
        return loss

    batches = dataUtils.batch_iter(zip(x_train, y_train), batch_size, n_epochs)
    copy_batches = dataUtils.batch_iter(zip(x_train, y_train), batch_size, n_epochs)

    # Training loop. For each batch...
    min_loss = float('Inf')
    best_at_step = 0
    total_step = sum(1 for _ in copy_batches)

    variables_names = [v.name for v in tf.trainable_variables()]
    print(variables_names) # ['embedding_layer/embedding_layer:0', 'W1:0', 'b1:0', 'W2:0', 'b2:0', 'Wh:0', 'bh:0', 'Wo:0']

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch, total_step)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            loss_val = val_step(x_val, y_val, writer=val_summary_writer)
            if loss_val <= min_loss:
                min_loss = loss_val
                best_at_step = current_step
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("")
        if current_step % checkpoint_every == 0:
            print('Best of valid = {}, at step {}'.format(min_loss, best_at_step))

    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print('Finish training. On test set:')
    loss = val_step(x_test, y_test, writer=None)
    print('loss is {}'.format(loss))

# end of print time cost
endtime = datetime.datetime.now()
print("Total cost {} seconds.".format((endtime - starttime).seconds))
