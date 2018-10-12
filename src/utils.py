# -*- coding:utf-8 -*-

import re
import jieba
import jieba.posseg

class IterDocument(object):
    """
    A class for reading large file memory-friendly
    """

    def __init__(self, path, sep=None):
        """
        :param path: path to the file
        :param sep: delimiter string between fields
        """
        self.path = path
        self.sep = sep

    def __iter__(self):
        """
        :return: iteration in lines
        """
        for line in open(self.path, 'r').readlines():
            line = line.strip().decode('utf-8')
            if line == '':
                continue
            if self.sep is not None:
                yield [item for item in line.split(self.sep) if item != ' ' and item != '']
            else:
                yield line


class TextCleaner:
    """
    A class for cleaning text
    """
    def __init__(self, punctuation=True, number=True, normalize=True):
        """
        :param punctuation: whether clean punctuation
        :param number: whether clean number
        :param normalize: whether normalize token
        """
        self.punc = punctuation
        self.num = number
        self.norm = normalize

        self.punctuation = IterDocument("../resource/punctuation")
    def clean(self, text):
        """
        :param text: the raw string
        :return: the string after cleaning
        """
        if self.punc:
            for p in self.punctuation:
                if p in text:
                    text = text.replace(p, " ")
                    #text = re.sub(p, " ", text)

        return text.strip()


class Segmentor:
    """
    A class for segmenting text
    """
    def __init__(self, user_dict=True):
        """
        :param user_dict: whether use user dict
        """
        self.seg = jieba
        self.seg_pos = jieba.posseg
        if user_dict:
            self.seg.load_userdict("../resource/userdict")
            self.seg.load_userdict("../resource/financial_complete.txt")
            self.seg.load_userdict("../resource/P2P_Online_loan_platform_2017.txt")
            self.seg.load_userdict("../resource/platform_name.txt")

    def seg_token(self, text):
        """
        :param text: the raw string
        :return: a list of token
        """
        return self.seg.lcut(text)

    def seg_token_pos(self, text):
        """
        :param text: the raw string
        :return: a list of token/pos
        """
        return ["%s/%s" % (token, pos) for token, pos in self.seg_pos.lcut(text)]


if __name__ == "__main__":
    tc = TextCleaner()
    seg = Segmentor()
    segfile = "../output/seg-text.txt"
    text = IterDocument("../data/comment")
    num = 0
    with open(segfile,"w") as f:
        for line in text:
            line = line.split('\t')[1]
            num += 1
            # print num
            # print line.encode('GBK')
            line = tc.clean(line)
            # print line
            segline = " ".join(seg.seg_token(line))
            # print segline
            f.write(segline.encode('UTF-8')+'\n')
            # if num==100:
            #     break

    