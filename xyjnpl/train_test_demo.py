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


def train_test():
    train = of.read_txt_and_deal(CONFIG.PATH_TRAIN)
    train = pre.hide_nr_demo(train)
    sent_train_deal = [x[2] for x in train]
    bags_train_deal = [x[3] for x in train]
    data, labels, tokenizer = cnn.fit_tokenizer(sent_train_deal, bags_train_deal)
    x_train, y_train, x_test, y_test = cnn.split_data(data, labels)
    # train_word2vec.word2vec_train(sent_train_deal)
    # data_test, labels_test = cnn.deal_data(tokenizer, x_test, y_test)
    model = cnn.fit_model(x_train, y_train, tokenizer)
    cnn.evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    train_test()
