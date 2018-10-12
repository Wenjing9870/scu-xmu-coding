# coding=utf-8
"""
Created on 2015-12-30 @author: Eastmount
"""

import time
import numpy as np
import heapq
import re
import os
import sys
import codecs
import shutil
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    corpus = []  # 文档语料 空格连接

    # 读取语料 一行语料为一个评论
    for line in open('C:/CCF/TF-IDF+Kmeans/data/segresult.txt', 'r',encoding='gb18030').readlines():
        # print(line)
        corpus.append(line.strip())
    #time.sleep(0)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i篇文本下的词频
    vectorizer = CountVectorizer()

    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    print(tfidf)
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    print(weight)
    print(len(weight))
    resName = "C:/CCF/TF-IDF+Kmeans/data/微贷网tfidf分析.txt"
    result = codecs.open(resName, 'w', 'utf-8')
    # for j in range(len(word)):
    #     result.write(word[j] + ' ')
    # result.write('\r\n\r\n')

    # 打印每个评论的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一篇文本下的词语权重
    # for i in range(len(weight)):
    #     print(u"-------这里输出第", i, u"条评论词语tf-idf权重------")
    #     for j in range(len(word)):
    #         result.write(str(weight[i][j]) + ' ')
    #     result.write('\r\n\r\n')

    #输出每条评论权重最高的几个词
    for i in range(len(weight)):
        Q = weight[i]
        sortLoc = np.argsort(-Q)
        sortVal = np.sort(-Q)
        print(sortLoc)
        print(sortVal)
        for j in range(5):
            h = sortLoc[j]
            result.write(word[h] + str(weight[i][h]) +'  ')

        # for j in range(len(F)):
        #     h = F(j)
        #     print(h)
        #     result.write(word[h] + str(weight[i][h]) + '  ')
        # for j in range(len(word)):
        #     if weight[i][j] != 0:
        #
        #         for m in range(len(word)):
        #             if weight[i][j] >= word[m]:
        #                 result.write(word[j] + str(weight[i][j]) + '  ')
        result.write('\r\n\r\n\r\n')

    #
    result.close()

    print('Start Kmeans:')
    from sklearn.cluster import KMeans

    result = codecs.open("C:/CCF/TF-IDF+Kmeans/data/kemeansresult.txt", 'w', 'utf-8')
    #for x in range(5,21):
       # clf = KMeans(n_clusters=10)
        #s=clf.fit(weight)
        #print(clf.inertia_)

    #print(clf.inertia_)
        # s = clf.fit(weight)
        #print(s)
        #
        # # 20个中心点
        #print(clf.cluster_centers_)
        #
        # # 每个样本所属的簇
        #print(clf.labels_)
        #i = 1
        #while i <= len(clf.labels_):
        #     print(i, clf.labels_[i - 1])
        #     i = i + 1
        #
        # # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
        # print(clf.inertia_)

    clf = KMeans(n_clusters=5)
    s = clf.fit(weight)
    print(s)
    print('类中心')
    #
    # # 20个中心点
    print(clf.cluster_centers_)
    #
    # # 每个样本所属的簇
    #lable = []
    print('每个样本所属的类')
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
         #lable[i][0]=i+1
         #lable[i][1]=clf.lable_[i]
         print(i, clf.labels_[i - 1])
         i = i + 1
         result.write(str(clf.labels_)+ ' ')
    #result.close()
    # # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数

    #输出每个类下的评论数据
    content = open('C:/CCF/DataSet/segresultt.txt', 'r',encoding='gb18030').readlines()
    j = 0
    while j <= 4:
        print('第'+str(j)+'类'+'\t')
        l = 1
        n = 0
        while l <= len(clf.labels_):
            if j == clf.labels_[l - 1]:
                print(l , content[l - 1])
                n = n + 1
            l = l + 1
        j = j + 1
        print('第'+str(j)+'类'+'有评论'+str(n)+'条')
    print(clf.inertia_)

