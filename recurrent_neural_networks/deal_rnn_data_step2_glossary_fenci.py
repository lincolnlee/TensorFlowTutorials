# -*- coding: UTF-8 -*-

from __future__ import print_function
#from __future__ import unicode_literals

import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 前一步生成的问答文件路径
train_encode_file = 'dataset3/train.enc'
train_decode_file = 'dataset3/train.dec'
test_encode_file = 'dataset3/test.enc'
test_decode_file = 'dataset3/test.dec'

print('开始创建词汇表...')
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# 参看tensorflow.models.rnn.translate.data_utils

vocabulary_size = 150000


# 生成词汇表文件
def gen_vocabulary_file(input_file, output_file):
    vocabulary = {}
    with open(input_file) as f:
        counter = 0
        for line in f:
            counter += 1
            word_generator = jieba.cut(line.strip().strip('“').strip('”'), cut_all=True)
            while True:
                try:
                    word = word_generator.next()
                    if len(word.strip()) > 0 and word.strip() != '':
                        print(word, end='')
                        if word.strip() in vocabulary:
                            vocabulary[word.strip()] += 1
                        else:
                            vocabulary[word.strip()] = 1
                    else:
                        break
                except StopIteration:
                    break
        vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
        # 取前150000个词
        if len(vocabulary_list) > vocabulary_size:
            vocabulary_list = vocabulary_list[:vocabulary_size]
        print(input_file + " 词汇表大小:", len(vocabulary_list))

        with open(output_file, "w") as ff:
            for word in vocabulary_list:
                print(word.strip())
                if len(word.strip()) and word.strip() != '' > 0:
                    ff.write(word.strip() + "\n")


gen_vocabulary_file(train_encode_file, "dataset3/train_encode_vocabulary")
gen_vocabulary_file(train_decode_file, "dataset3/train_decode_vocabulary")
