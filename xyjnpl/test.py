# -*- coding: utf-8 -*-

"""
用于演示
本项目用于2019年毕设，使用CNN对《西游记》小说中的人物关系进行识别
"""
import sys
import os

# 当前项目路径加入到环境变量中，让解析器能找到第一model的目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

import xyjnpl.openfile as of
import config.setting as CONFIG
import xyjnpl.cnn as cnn
import xyjnpl.preprocessing as pre
import jieba
import pickle

CONFIG.VERSION = 'main'


def train_test():
    train = of.read_txt_and_deal(CONFIG.PATH_TEST_DEAL)
    train = pre.hide_nr_demo(train)
    sent_train_deal = list()

    for s in train:
        nr1 = s[0]
        nr2 = s[1]
        x = s[2]
        x = pre.hide_nr(x, nr1, nr2)
        words = jieba.lcut(x)
        # word_str = ' '.join(words)
        sent_train_deal.append(words)

    bags_train_deal = [int(x[3]) for x in train]

    with open('model/tokenizer_' + str(CONFIG.VERSION) + '.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    data_test, labels_test = cnn.deal_data(tokenizer, sent_train_deal, bags_train_deal)
    model = cnn.load_models()
    cnn.evaluate_model(model, data_test, labels_test)


if __name__ == '__main__':
    train_test()
