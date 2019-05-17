# -*- coding: utf-8 -*-
import os
import sys

sys.path.append('/home/ml/lpagec/tensorflow/FM-GAN')

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb

from model2 import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, _clip_gradients_seperate_norm
from denoise import *

logging.set_verbosity(logging.INFO)
flags = tf.app.flags
FLAGS = flags.FLAGS

PATH_TO_SAVE = sys.argv[1]
TEMPERATURE  = float(sys.argv[2])
STD = TEMPERATURE

class Options(object):
    def __init__(self):
        self.dis_steps = 1
        self.gen_steps = 5
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' #'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 51
        self.n_words = 5728
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2
        self.embed_size = 300
        self.latent_size = 128
        self.lr = 1e-5

        self.rnn_share_emb = True
        self.additive_noise_lambda = 0.0
        self.bp_truncation = None
        self.n_hid = 100

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 1000
        self.max_epochs = 60
        self.n_gan = 128  # self.filter_size * 3
        self.L = 100 #TEMPERATURE# 100 * TEMPERATURE
        print('in temp', self.L)

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None #None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.relu_w = False

        self.save_path = "./save_news/" + "news_" + str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(self.permutation)
        self.log_path = "./log"
        self.print_freq = 10
        self.valid_freq = 100

        # batch norm & dropout
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1

        self.discrimination = False
        self.H_dis = 300
        self.ef_dim = 128
        self.sigma_range = [2]

        self.epsilon = 100
        self.niter = 20

        self.sent_len = self.maxlen + 2*(self.filter_shape-1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        self.sentence = self.maxlen - 1
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def discriminator(x, opt, prefix = 'd_', is_prob = False, is_reuse = None):
    W_norm_d = embedding_only(opt, prefix = prefix, is_reuse = is_reuse)   # V E
    H = encoder(x, W_norm_d, opt, prefix = prefix + 'enc_',  is_prob=is_prob, is_reuse = is_reuse)
    logits = discriminator_2layer(H, opt, is_reuse = is_reuse)

    return logits, H

def encoder(x, W_norm_d, opt, prefix = 'd_', is_prob = False, is_reuse = None, is_padded = True):
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2],[0]])
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)   # batch L emb
    if not is_padded:  # pad the input with pad_emb
        pad_emb = tf.expand_dims(tf.expand_dims(W_norm_d[0],0),0) # 1*v
        x_emb = tf.concat([tf.tile(pad_emb, [opt.batch_size, opt.filter_shape-1, 1]), x_emb],1)

    x_emb = tf.expand_dims(x_emb,3)   # batch L emb 1
    #bp()
    if opt.layer == 3:
        H = conv_model_3layer(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    else:
        H = conv_model(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    return tf.squeeze(H)



def textGAN(x, x_org, opt):
    # print x.get_shape()  # batch L
    res_ = {}

    with tf.variable_scope("pretrain"):
        z = tf.random_normal([opt.batch_size, opt.latent_size], mean=0., stddev=STD)
        _, W_norm = embedding(x, opt, is_reuse = None)
        _, syn_sent, logits = lstm_decoder_embedding(z, x_org, W_norm, opt, feed_previous=True, is_sampling=True)
        prob = [tf.nn.softmax(l*opt.L) for l in logits]
        prob = tf.stack(prob,1)

    with tf.variable_scope("d_net"):
        logits_real, H_real = discriminator(x, opt)


    with tf.variable_scope("d_net"):
        logits_fake, H_fake = discriminator(prob, opt, is_prob = True, is_reuse = True)

    res_['syn_sent'] = syn_sent
    res_['real_f'] = tf.squeeze(H_real)
    # Loss

    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake), logits = logits_fake))

    fake_mean = tf.reduce_mean(H_fake,axis=0)
    real_mean = tf.reduce_mean(H_real,axis=0)
    mean_dist = tf.sqrt(tf.reduce_mean((fake_mean - real_mean)**2))
    res_['mean_dist'] = mean_dist

    GAN_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
    MMD_loss = compute_MMD_loss(tf.squeeze(H_fake), tf.squeeze(H_real), opt)

    # SINK_loss = sinkhorn_normalized(tf.squeeze(H_fake), tf.squeeze(H_real), opt)
    SINK_loss = IPOT_distance(tf.squeeze(H_fake), tf.squeeze(H_real), opt)

    G_loss = SINK_loss
    D_loss =  SINK_loss * (-1)

    res_['mmd'] = MMD_loss
    res_['sinkhorn'] = SINK_loss
    res_['gan'] = tf.reduce_mean(GAN_loss)
    # *tf.cast(tf.not_equal(x_temp,0), tf.float32)
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    summaries = [
                "learning_rate",
                "G_loss",
                "D_loss"
                # "gradients",
                # "gradient_norm",
                ]
    global_step = tf.Variable(0, trainable=False)

    all_vars = tf.trainable_variables()
    g_vars = [var for var in all_vars if
                  var.name.startswith('pretrain')]
    d_vars = [var for var in all_vars if
              var.name.startswith('d_')]
    print([g.name for g in g_vars])

    ''' Update Operators '''

    return res_, G_loss, D_loss, None, None


def run_model(opt, train, val, ixtoword):

    try:
        params = np.load('./param_g.npz')
        if params['Wemb'].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            opt.W_emb = params['Wemb']
        else:
            print('Emb Dimension mismatch: param_g.npz:'+ str(params['Wemb'].shape) + ' opt: ' + str((opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_org_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        res_, g_loss_, d_loss_, gen_op, dis_op = textGAN(x_, x_org_, opt)
        merged = tf.summary.merge_all()

    uidx = 0
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=tf.GraphOptions(build_cost_model=1))
    #config = tf.ConfigProto(device_count={'GPU':0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)

    # keep all checkpoints
    saver = tf.train.Saver(max_to_keep=None)

    run_metadata = tf.RunMetadata()

    with tf.Session(config = config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                #pdb.set_trace()

                t_vars = tf.trainable_variables()
                #print([var.name[:-2] for var in t_vars])
                loader = restore_from_save(t_vars, sess, opt)
                print('\nload successfully\n')

            except Exception as e:
                print(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())


        ''' validation '''
        valid_index = np.random.choice(len(val), opt.batch_size)
        val_sents = [val[t] for t in valid_index]

        val_sents_permutated = add_noise(val_sents, opt)

        x_val_batch = prepare_data_for_cnn(val_sents_permutated, opt)
        x_val_batch_org = prepare_data_for_rnn(val_sents, opt)

        d_loss_val = sess.run(d_loss_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org})
        g_loss_val = sess.run(g_loss_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org})


        res = sess.run(res_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org})
        try:
            print("Validation d_loss %f, g_loss %f  mean_dist %f" % (d_loss_val, g_loss_val, res['mean_dist']))
            print("Sent:" + u' '.join([ixtoword[x] for x in res['syn_sent']
                                   [0] if x != 0]))#.encode('utf-8', 'ignore').decode("utf8").strip())
            print("MMD loss %f, GAN loss %f" % (res['mmd'], res['gan']))
        except Exception as e: 
            print(e)

        # np.savetxt('./text_arxiv/syn_val_words.txt', res['syn_sent'], fmt='%i', delimiter=' ')
        if opt.discrimination:
            print ("Real Prob %f Fake Prob %f" % (res['prob_r'], res['prob_f']))

        for i in range(268590 // 1000 + 1): # generate 10k sentences # generate 268590
            valid_index = np.random.choice(
                len(val), opt.batch_size)
            val_sents = [val[t] for t in valid_index]
            val_sents_permutated = add_noise(val_sents, opt)
            x_val_batch = prepare_data_for_cnn(
                val_sents_permutated, opt)
            x_val_batch_org = prepare_data_for_rnn(val_sents, opt)
            res = sess.run(res_, feed_dict={
                           x_: x_val_batch, x_org_: x_val_batch_org})
            if i == 0:
                valid_text = res['syn_sent']
            else:
                valid_text = np.concatenate(
                    (valid_text, res['syn_sent']), 0)

        valid_text = valid_text[:268590]
        np.savetxt(PATH_TO_SAVE,valid_text, fmt='%i', delimiter=' ')
        print('saved!\n\n\n') 
        exit()
        
        val_set = [prepare_for_bleu(s) for s in val] #val_sents]
        bleu_prepared = [prepare_for_bleu(s) for s in res['syn_sent']]
        for i in range(len(val_set) // opt.batch_size):
            batch = val_set[i*opt.batch_size: (i+1) * opt.batch_size]
            [bleu2s, bleu3s, bleu4s] = cal_BLEU(bleu_prepared, {0: batch}) #val_set})
            print('Val BLEU (2,3,4): ' + ' '.join([str(round(it, 3)) for it in (bleu2s, bleu3s, bleu4s)]))


def main():
    trainpath = "./data/NewsData/train_news.txt"
    testpath = "./data/NewsData/test_news.txt"
    train, val =  np.loadtxt(trainpath), np.loadtxt(testpath)
    ixtoword, _ = cPickle.load(open('./data/NewsData/vocab_news.pkl','rb'))
    ixtoword = {i:x for i,x in enumerate(ixtoword)}
    opt = Options()

    print(dict(opt))
    print('Total words: %d' % opt.n_words)
    run_model(opt, train, val, ixtoword)


if __name__ == '__main__':
    main()
