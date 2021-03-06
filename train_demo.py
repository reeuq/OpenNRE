# -*- coding: utf-8 -*-

import nrekit
import numpy as np
import tensorflow as tf
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
dataset_name = 'nyt'
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
dataset_dir = os.path.join('./data', dataset_name)
if not os.path.isdir(dataset_dir):
    raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

# The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                                        os.path.join(dataset_dir, 'word_vec.json'),
                                                        os.path.join(dataset_dir, 'rel2id.json'), 
                                                        mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                        shuffle=True)
test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'), 
                                                       os.path.join(dataset_dir, 'word_vec.json'),
                                                       os.path.join(dataset_dir, 'rel2id.json'), 
                                                       mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                       shuffle=False)

framework = nrekit.framework.re_framework(train_loader, test_loader)

class model(nrekit.framework.re_model):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
        
        # Embedding
        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)
        x_sdp = nrekit.network.embedding.word_embedding(self.sdp, self.word_vec_mat)

        # Encoder
        if model.encoder == "pcnn":
            x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
        elif model.encoder == "cnn":
            x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
            x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
        elif model.encoder == "rnn":
            x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
        elif model.encoder == "birnn":
            x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
        elif model.encoder == "capsnn":
            x_train = nrekit.network.encoder.capsnn(x, keep_prob=0.5)
            x_test = nrekit.network.encoder.capsnn(x, keep_prob=1.0)
        elif model.encoder == "pcnn_cnn":
            x_train_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            x_train_sdp = nrekit.network.encoder.cnn(x_sdp, keep_prob=0.5)
            x_test_sdp = nrekit.network.encoder.cnn(x_sdp, keep_prob=1.0)
            x_train = tf.concat([x_train_word, x_train_sdp], 1)
            x_test = tf.concat([x_test_word, x_test_sdp], 1)
        elif model.encoder == "pcnn_cnn_full":
            x_train_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            x_train_sdp = nrekit.network.encoder.cnn(x_sdp, keep_prob=0.5)
            x_test_sdp = nrekit.network.encoder.cnn(x_sdp, keep_prob=1.0)
            x_train = tf.concat([x_train_word, x_train_sdp], 1)
            x_test = tf.concat([x_test_word, x_test_sdp], 1)
            with tf.variable_scope('full-connect', reuse=tf.AUTO_REUSE):
                relation_matrix = tf.get_variable('weight_matrix', shape=[x_train.shape[1], 230], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable('weight_bias', shape=[230], dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
                x_train = tf.nn.tanh(tf.matmul(x_train, relation_matrix) + bias)
                x_test = tf.nn.tanh(tf.matmul(x_test, relation_matrix) + bias)
        elif model.encoder == "pcnn_rnn":
            x_train_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            x_train_sdp = nrekit.network.encoder.rnn(x_sdp, self.sdp_length, keep_prob=0.5)
            x_test_sdp = nrekit.network.encoder.rnn(x_sdp, self.sdp_length, keep_prob=1.0)
            x_train = tf.concat([x_train_word, x_train_sdp], 1)
            x_test = tf.concat([x_test_word, x_test_sdp], 1)
        elif model.encoder == "pcnn_birnn":
            x_train_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            x_train_sdp = nrekit.network.encoder.birnn(x_sdp, self.sdp_length, keep_prob=0.5)
            x_test_sdp = nrekit.network.encoder.birnn(x_sdp, self.sdp_length, keep_prob=1.0)
            x_train = tf.concat([x_train_word, x_train_sdp], 1)
            x_test = tf.concat([x_test_word, x_test_sdp], 1)
        elif model.encoder == "birnn_att_cnn":
            x_word = nrekit.network.encoder.birnn_att(x, self.length, hidden_size=110, keep_prob=1.0)
            x_sdp = nrekit.network.encoder.cnn(x_sdp, hidden_size=220, keep_prob=1.0)
            with tf.variable_scope("sentence_sdp_att", reuse=tf.AUTO_REUSE):
                attention_score = tf.transpose(tf.nn.softmax(tf.matmul(tf.expand_dims(x_sdp, 1), tf.transpose(x_word, perm=[0, 2, 1])), -1), perm=[0, 2, 1])
                x_att = tf.squeeze(tf.matmul(tf.transpose(x_word, perm=[0, 2, 1]), attention_score), [-1])
            x_train = tf.contrib.layers.dropout(x_att, keep_prob=0.5)
            x_test = tf.contrib.layers.dropout(x_att, keep_prob=1.0)
        elif model.encoder == "birnn_rnn":
            x_train_word = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
            x_test_word = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
            x_train_sdp = nrekit.network.encoder.rnn(x_sdp, self.sdp_length, keep_prob=0.5)
            x_test_sdp = nrekit.network.encoder.rnn(x_sdp, self.sdp_length, keep_prob=1.0)
            x_train = tf.concat([x_train_word, x_train_sdp], 1)
            x_test = tf.concat([x_test_word, x_test_sdp], 1)
        # elif model.encoder == "birnn_att_concat_cnn":
        #     x_word = nrekit.network.encoder.birnn_att(x, self.length, hidden_size=110, keep_prob=1.0)
        #     x_sdp = nrekit.network.encoder.cnn(x_sdp, hidden_size=220, keep_prob=1.0)
        #     with tf.variable_scope("sentence_sdp_att", reuse=tf.AUTO_REUSE):
        #         attention_w = tf.get_variable('attention_w', shape=[1, x.shape[1]], dtype=tf.float32,
        #                                           initializer=tf.contrib.layers.xavier_initializer())
        #
        #         attention_score = tf.transpose(tf.nn.softmax(tf.matmul(tf.expand_dims(x_sdp, 1), tf.transpose(x_word, perm=[0, 2, 1])), -1), perm=[0, 2, 1])
        #         x_att = tf.squeeze(tf.matmul(tf.transpose(x_word, perm=[0, 2, 1]), attention_score), [-1])
        #     x_train = tf.contrib.layers.dropout(x_att, keep_prob=0.5)
        #     x_test = tf.contrib.layers.dropout(x_att, keep_prob=1.0)
        elif model.encoder == "none_birnn":
            x_train = nrekit.network.encoder.birnn(x_sdp, self.sdp_length, keep_prob=0.5)
            x_test = nrekit.network.encoder.birnn(x_sdp, self.sdp_length, keep_prob=1.0)
        elif model.encoder == "none_cnn":
            x_train = nrekit.network.encoder.cnn(x_sdp, keep_prob=0.5)
            x_test = nrekit.network.encoder.cnn(x_sdp, keep_prob=1.0)
        else:
            raise NotImplementedError

        # Selector
        if model.selector == "att":  # 为什么att的test_logit不进行softmax？？
            self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
        elif model.selector == "ave":
            self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        elif model.selector == "one":
            self._train_logit, train_repre = nrekit.network.selector.bag_one(x_train, self.scope, self.label, self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_one(x_test, self.scope, self.label, self.rel_tot, False, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        elif model.selector == "cross_max":
            self._train_logit, train_repre = nrekit.network.selector.bag_cross_max(x_train, self.scope, self.rel_tot, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_cross_max(x_test, self.scope, self.rel_tot, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        else:
            raise NotImplementedError
        
        # Classifier
        self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot, weights_table=self.get_weights())
 
    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0 
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
            print("Finish calculating")
        return weights_table

use_rl = False
if len(sys.argv) > 2:
    model.encoder = sys.argv[2]
if len(sys.argv) > 3:
    model.selector = sys.argv[3]
if len(sys.argv) > 4:
    if sys.argv[4] == 'rl':
        use_rl = True

if use_rl:
    rl_framework = nrekit.rl.rl_re_framework(train_loader, test_loader)
    rl_framework.train(model, nrekit.rl.policy_agent, model_name=dataset_name + "_" + model.encoder + "_" + model.selector + "_rl", max_epoch=60, ckpt_dir="checkpoint")
else:
    framework.train(model, model_name=dataset_name + "_" + model.encoder + "_" + model.selector, max_epoch=60, ckpt_dir="checkpoint", gpu_nums=1)
