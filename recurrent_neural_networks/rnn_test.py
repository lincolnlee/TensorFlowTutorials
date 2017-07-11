# -*- coding: UTF-8 -*-


from __future__ import print_function

import tensorflow as tf  # 0.12
from tensorflow.python.models.tutorials.rnn.translate import seq2seq_model
import os
import numpy as np
import jieba
import random

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vocabulary = 'dataset3/train_encode_vocabulary'
train_decode_vocabulary = 'dataset3/train_decode_vocabulary'


def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip().decode('utf-8') for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab


vocab_en, _, = read_vocabulary(train_encode_vocabulary)
_, vocab_de, = read_vocabulary(train_decode_vocabulary)

# 词汇表大小5000
vocabulary_encode_size = 135890
vocabulary_decode_size = 137423

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256  # 每层大小
num_layers = 3  # 层数
batch_size = 1

model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_encode_size,
                                   target_vocab_size=vocabulary_decode_size,
                                   buckets=buckets,
                                   size=layer_size,
                                   num_layers=num_layers,
                                   max_gradient_norm=5.0,
                                   batch_size=batch_size,
                                   learning_rate=0.48,
                                   learning_rate_decay_factor=0.97,
                                   use_lstm=True,
                                   forward_only=True)
model.batch_size = 1

with tf.Session() as sess:
    # 恢复前一次训练
    ckpt = tf.train.get_checkpoint_state('model/')
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("没找到模型")

    while True:
        input_string = raw_input('me > ')
        # 退出
        if input_string == 'quit':
            exit()

        input_string_vec = []
        word_generator = jieba.cut(input_string.strip(), cut_all=False)
        while True:
            try:
                word = word_generator.next()
                if len(word.strip()) > 0 and word.strip() != '':
                    input_string_vec.append(vocab_en.get(word, UNK_ID))
                else:
                    break
            except StopIteration:
                break

        bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]},
                                                                         bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]

        response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs]).strip('__UNK__')
        if len(response) == 0 or response == '':
            rdm_num = random.sample(xrange(2), 1)
            if rdm_num[0] == 1:
                response = "我竟无言以对"
            else:
                response = "这话我没法接"
        print('AI > ' + response)

