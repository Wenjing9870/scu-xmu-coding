# -*- coding:utf-8 -*-

from __future__ import print_function

import argparse
import utils
import pandas as pd

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import PreProcess, IterDocument

logger = utils.get_logger("clustering")


def make_stopwords(stop_words_path=None):
    if stop_words_path is None:
        logger.info("No stopwords.")
        return None
    logger.info("Load stopwords from %s" % stop_words_path)
    lines = IterDocument(stop_words_path)
    stop_words = " ".join(lines)
    logger.info("Stop words: %s" % stop_words)
    return stop_words.split(" ")


def clustering(text, n, stop):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000,
                                       min_df=2, stop_words=stop,
                                       use_idf=True)
    x = tfidf_vectorizer.fit_transform(text)

    km = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=1, verbose=True)

    logger.info("Clustering for %s class" % n)
    pred = km.fit_predict(x)
    res = pd.DataFrame([text, pred]).T
    res.columns = ["text", "class"]

    return res


def keyword_discovery(clusters, n):
    for i in range(n):
        counter = Counter()
        df = clusters[clusters["class"] == i]
        logger.info("The %s class, size is: %s" % (i, df.shape[0]))
        for idx, row in df.iterrows():
            counter.update(row["text"].split(" "))
        sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        top_words = [item[0] for item in sorted_words[0:20]]
        logger.info("The %s class, top 20 words: %s" % (i, " ".join(top_words)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to comment data')
    parser.add_argument('-n', dest='num_cluster', default=10, help='number of cluster')
    parser.add_argument('-s', dest='stopwords', default=None, help='path to stopwords')

    args = parser.parse_args()

    data = PreProcess(args.input_file).make_data_set()
    stopwords = make_stopwords(args.stopwords)
    cluster = clustering(data.text_seg, args.num_cluster, stopwords)

    print(cluster.head(20))

    # keyword_discovery(cluster, args.num_cluster)
