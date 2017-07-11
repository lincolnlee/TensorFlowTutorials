# -*- coding: UTF-8 -*-

from __future__ import print_function
#from __future__ import unicode_literals

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

vocabulary_size = 5000


# 生成词汇表文件
def gen_vocabulary_file(input_file, output_file):
    vocabulary = {}
    with open(input_file) as f:
        counter = 0
        for line in f:
            counter += 1
            for word in line.decode('utf-8'):
                if u'\u4e00' < word < u'\u9fa5' \
                        or u'\uff00' < word < u'\uffef' \
                        or u'\u0030' < word < u'\u0039' \
                        or u'\u0061' < word < u'\u007a' \
                        or u'\u0041' < word < u'\u005a':
                    # 中文or全角ASCII、全角中英文标点、半宽片假名、半宽平假名、半宽韩文字母or数字or小写字母or大写字母
                    if word in vocabulary:
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
        vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
        # 取前5000个常用汉字, 应该差不多够用了
        if len(vocabulary_list) > vocabulary_size:
            vocabulary_list = vocabulary_list[:5000]
        print(input_file + " 词汇表大小:", len(vocabulary_list))

        with open(output_file, "w") as ff:
            for word in vocabulary_list:
                print(word)
                ff.write(word + "\n")


gen_vocabulary_file(train_encode_file, "dataset3/train_encode_vocabulary")
gen_vocabulary_file(train_decode_file, "dataset3/train_decode_vocabulary")