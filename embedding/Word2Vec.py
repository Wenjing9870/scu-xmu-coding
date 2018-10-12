# -*- coding:utf-8 -*-

# from gensim import models


# filePath = '../../data/comment.txt'
# file = open(filePath)
# line = file.readline()
# count = 0
# while line:
#     print line
#     line = file.readline()
#     count += 1
#     if count == 10:
#         break

# sentences = models.word2vec.LineSentence(filePath, limit=10)
# model = models.word2vec.Word2Vec(sentences, size=100, window=2, sg=1, min_count=5, workers=4)

import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import xrange
import random
import csv
from sklearn.manifold import TSNE


print('====================Word2Vec====================')


# 读取源文件, 并转为list的词输出
def read_words(filename):
    words = []
    with open(filename, 'r') as fp:
        for line0 in fp.readlines():
            line0 = line0.strip()
            line = line0.split(' ')
            for word in line:
                words.append(word)
    return words


words = read_words('../../data/comment.txt')
# print(words, np.shape(words), type(words))

# count = [['UNK', -1]]  # 初始化单词频数统计集合

'''
1.给@param “words”中出现过的单词做频数统计，取top 1999频数的单词放入dictionary中，以便快速查询。
2.给哈利波特这本“单词库”@param“words”编码，出现在top 1999之外的单词，统一令其为“UNK”（未知），编号为0，并统计这些单词的数量。
@return: 哈利波特这本书的编码data，每个单词的频数统计count，词汇表dictionary及其反转形式reverse_dictionary
'''


def build_dataset(words):
    # most_common方法： 去top2000的频数的单词，创建一个dict,放进去。以词频排序
    counter = collections.Counter(words).most_common()  # length of all counter:22159 取top1999频数的单词作为vocabulary，其他的作为unknown
    # print('123  :', type(counter))  # list类型
    # print(len(counter))  # 1999个
    # print(counter)
    # [('the', 51897), ('and', 27525), ('to', 26996), ('he', 22203), ('of', 21851), ('a', 21043), ('harry', 18165), ('was', 15637), ('you', 14627), ('it', 14489),。 ('member', 49), ('fake', 49)。]
    # count.extend(counter)
    # 搭建dictionary
    dictionary = {}
    for word, _ in counter:
        dictionary[word] = len(dictionary)
    # print('234   :',dictionary.get('the'))   输出为1
    data = []
    # 全部单词转为编号
    # 先判断这个单词是否出现在dictionary，如果是，就转成编号，如果不是，则转为编号0（代表UNK）
    for word in words:
        index = dictionary[word]
        data.append(index)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # print('aaaaa   ', data)                 # 根据文章，将全文每个数，对应的排名标记， 比如 the 出现最多，则对应的数为1，不在前1999的标记为0  【959, 18, 7, 6, 1728, 0, 306, 13, 450, 1175, 32, 180, 265, 3, 9, 205,】
    # print('00000   ', count[300])               # 前1999，个对应的出现的次数，第一个数是 总共多少不在前1999的字数   比如： the:51897
    # print('bbbbb   ', dictionary)          # 前1999的数的dict  ['the':1]这种
    # print('1111   :', reverse_dictionary)  # 按照dict值来排序，变得有规律起来。{0: 'UNK', 1: 'the', 2: 'and', 3: 'to', 4: 'he', 5: 'of', 6: 'a', 7: 'harry', 8: 'was', 9:}
    return data, counter, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # 删除原始单词列表，节约内存
# print(data, count, dictionary, reverse_dictionary)
vocabulary_size = len(count)  # 预定义频繁单词库的长度

data_index = 0

'''
采用Skip-Gram模式,  生成我们需要的 input label 的那种格式。，可以参看第二个小例子。
生成word2vec训练样本
@param batch_size:每个批次训练多少样本
@param num_skips: 为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
@param skip_window:单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips
'''


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# 开始训练
batch_size = 128
embedding_size = 100
skip_window = 2
num_skips = 2
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量
# 验证数据
valid_size = 10  # 抽取的验证单词数
valid_window = 50  # 验证单词只从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 不重复在0——10l里取16个


graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # 单词维度为 2000单词大小，128向量维度
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))   # 初始化embedding vector
    # 使用tf.nn.embedding_lookup查找输入train_inputs 对应的向量embed
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # 用NCE loss作为优化训练的目标
    # tf.truncated_normal初始化nce loss中的权重参数 nce_weights,并将nce_biases初始化为0
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # 计算学习出的词向量embedding在训练数据上的loss,并使用tf.reduce_mean进行魂汇总
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
    # 学习率为1.0，L2范式标准化后的enormalized_embedding。
    # 通过cos方式来测试  两个之间的相似性，与向量的长度没有关系。
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    normalized_embeddings = embeddings / norm   # 除以其L2范数后得到标准化后的normalized_embeddings
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)    # 如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)   # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
    print '相似性：', similarity
    init = tf.global_variables_initializer()

num_steps = 10001
loss_list = []
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    avg_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)  # 产生批次训练样本
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}   # 赋值
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        loss_list.append(loss_val)
        avg_loss += loss_val
        # 每2000次，计算一下平均loss并显示出来。
        if step % 2000 == 0:
            if step > 0:
                avg_loss /= 2000
            print "Avg loss at step ", step, ": ", avg_loss
            avg_loss = 0
            # 每10000次，验证单词与全部单词的相似度，并将与每个验证单词最相似的8个找出来。
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]  # 得到验证单词
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]     # 每一个valid_example相似度最高的top-k个单词
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embedding = normalized_embeddings.eval()
    print 'Final Embedding Matrix:', final_embedding, np.shape(final_embedding)


def save_embeddings(filepath):
    csvfile = open(filepath, 'w')  # 设置为utf-8在csv文件中中文乱码但读取正常
    wrtr = csv.writer(csvfile)
    line = []
    for i in range(len(final_embedding)):
        word = reverse_dictionary[i]
        vector = []
        for j in range(len(final_embedding[i])):
            vector.append(final_embedding[i][j])
        line.append([word] + vector)
    for x in line:
        wrtr.writerow(x)


embeddings_save_path = '../../data/Word2Vec_embeddings(100,2,2).csv'
save_embeddings(embeddings_save_path)


# 画损失迭代图
def plot_loss(num_steps, loss_list):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(num_steps), loss_list)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Loss")
    plt.grid(True)
    plt.savefig('../../data/Word2Vec_loss(100,2,2).eps')


# 可视化Word2Vec散点图并保存
def plot_with_labels(low_dim_embs, labels, filename):
    # low_dim_embs 降维到2维的单词的空间向量
    assert low_dim_embs.shape[0] >= len(labels), "more labels than embedding"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        # 展示单词本身
        plt.rcParams['font.sans-serif'] = ['simhei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.rcParams['font.family'] = ['simhei']
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


plot_loss(num_steps, loss_list)
# ValueError: matplotlib display text must have all code points < 128 or use Unicode strings
# tsne实现降维，将原始的嵌入向量降到2维
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=5000)
plot_number = 300
low_dim_embs = tsne.fit_transform(final_embedding[:plot_number, :])
labels = [reverse_dictionary[i] for i in range(plot_number)]
plot_with_labels(low_dim_embs, labels, '../../data/Word2Vec_word(100,2,2).eps')
