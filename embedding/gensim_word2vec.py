# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def get_sentences(filePath):
    for line in open(filePath):
        line = line.strip()
        words = line.split(' ')
        if line:
            yield words


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def gensim_model(sentences, embed_size, logPath):
    model = Word2Vec(sentences, size=embed_size, window=2, sg=1, workers=4, iter=10, compute_loss=True)
    print 'Loss:', model.get_latest_training_loss()
    vocab_size = len(model.wv.vocab)
    # project part of vocab
    embedding_matrix = np.zeros((vocab_size, embed_size))
    with open(os.path.join(logPath, 'metadata.tsv'), 'w+') as file_metadata:
        for i, word in enumerate(model.wv.index2word[:vocab_size]):
            embedding_matrix[i] = model[word]
            file_metadata.write(word + '\n')
    # print embedding_matrix[5320], type(embedding_matrix[5320])

    # define the model without training
    sess = tf.InteractiveSession()
    with tf.device("/cpu:0"):
        tf.Variable(embedding_matrix, trainable=False, name='prefix_embedding')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logPath, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'prefix_embedding'
    embed.metadata_path = os.path.join(logPath, 'metadata.tsv')

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)

    saver.save(sess, os.path.join(logPath, 'model.ckpt'), global_step=10000)
    return vocab_size

    # open tensorboard with logdir, check localhost:6006 for viewing your embedding.
    # tensorboard --logdir="../../data/projector/"


if __name__ == '__main__':
    fileinPath = sys.argv[1]  # 原始分词后评论数据s所在文件夹
    logPath = sys.argv[2]  # 输出的日志文件夹路径
    sentences = MySentences(fileinPath)
    # sentences = get_sentences(fileinPath)
    vocab_size = gensim_model(sentences, 100, logPath)
    print "Vocabulary size:", vocab_size
