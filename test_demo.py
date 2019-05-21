import nrekit
import numpy as np
import tensorflow as tf
import sys
import os
import json
import codecs


def positive_evaluation(predict_results):
    predict_y = predict_results[0]
    predict_y_prob = predict_results[1]
    y_given = predict_results[2]

    positive_num = 0
    #find the number of positive examples
    for yi in range(y_given.shape[0]):
        if y_given[yi, 0] > 0:
            positive_num += 1
    # if positive_num == 0:
    #     positive_num = 1
    # sort prob
    index = np.argsort(predict_y_prob)[::-1]

    all_pre = [0]
    all_rec = [0]
    p_n = 0
    p_p = 0
    n_p = 0
    # print y_given.shape[0]
    for i in range(y_given.shape[0]):
        labels = y_given[index[i],:] # key given labels
        py = predict_y[index[i]] # answer
        if labels[0] == 0:
            # NA bag
            if py > 0:
                n_p += 1
        else:
            # positive bag
            if py == 0:
                p_n += 1
            else:
                flag = False
                for j in range(y_given.shape[1]):
                    if j == -1:
                        break
                    if py == labels[j]:
                        flag = True # true positive
                        break
                if flag:
                    p_p += 1
        if (p_p+n_p) == 0:
            precision = 1
        else:
            precision = float(p_p)/(p_p+n_p)
        recall = float(p_p)/positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    return [all_pre[1:], all_rec[1:]]


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
        elif model.encoder == "pcnn_cnn":
            x_train_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test_word = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            x_train_sdp = nrekit.network.encoder.cnn(x_sdp, keep_prob=0.5)
            x_test_sdp = nrekit.network.encoder.cnn(x_sdp, keep_prob=1.0)
            x_train = tf.concat([x_train_word, x_train_sdp], 1)
            x_test = tf.concat([x_test_word, x_test_sdp], 1)
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
                attention_score = tf.transpose(
                    tf.nn.softmax(tf.matmul(tf.expand_dims(x_sdp, 1), tf.transpose(x_word, perm=[0, 2, 1])), -1),
                    perm=[0, 2, 1])
                x_att = tf.squeeze(tf.matmul(tf.transpose(x_word, perm=[0, 2, 1]), attention_score), [-1])
            x_train = tf.contrib.layers.dropout(x_att, keep_prob=0.5)
            x_test = tf.contrib.layers.dropout(x_att, keep_prob=1.0)
        elif model.encoder == "none_birnn":
            x_train = nrekit.network.encoder.birnn(x_sdp, self.sdp_length, keep_prob=0.5)
            x_test = nrekit.network.encoder.birnn(x_sdp, self.sdp_length, keep_prob=1.0)
        elif model.encoder == "none_cnn":
            x_train = nrekit.network.encoder.cnn(x_sdp, keep_prob=0.5)
            x_test = nrekit.network.encoder.cnn(x_sdp, keep_prob=1.0)
        else:
            raise NotImplementedError

        # Selector
        if model.selector == "att":
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
        elif model.selector == "max":
            self._train_logit, train_repre = nrekit.network.selector.bag_maximum(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_maximum(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
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

if len(sys.argv) > 2:
    model.encoder = sys.argv[2]
if len(sys.argv) > 3:
    model.selector = sys.argv[3]

auc, pred_result, predict_y, predict_y_prob, y_given = framework.test(model, ckpt="./checkpoint/" + dataset_name + "_" + model.encoder + "_" + model.selector + '1', return_result=True)

max_len = 0
for y_instance in y_given:
    if len(y_instance) > max_len:
        max_len = len(y_instance)
for y_instance in y_given:
    for i in range(max_len-len(y_instance)):
        y_instance.append(-1)
y_given = np.array(y_given)
test_pr = positive_evaluation((predict_y, predict_y_prob, y_given))

total = 0
correct_num = 0
for i in range(len(predict_y)):
    if y_given[i][0] != 0:
        total += 1
        if predict_y[i] in y_given[i]:
            correct_num += 1
accuracy = correct_num / total
print("accuracy: %f" % accuracy)

with codecs.open('./test_result_Zeng/' + dataset_name + "_" + model.encoder + "_" + model.selector + '1' + ".txt", 'w',
                 encoding='utf-8') as outfile:
    all_pre = test_pr[0]
    all_rec = test_pr[1]
    for i, p in enumerate(all_pre):
        outfile.write(str(p) + ' ' + str(all_rec[i]) + '\n')

with codecs.open('./test_result/' + dataset_name + "_" + model.encoder + "_" + model.selector + '1' + "_pred.json", 'w', encoding='utf-8') as outfile:
    json.dump(pred_result, outfile)

