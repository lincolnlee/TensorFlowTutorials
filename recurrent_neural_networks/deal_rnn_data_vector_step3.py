# -*- coding: UTF-8 -*-

# 前一步生成的问答文件路径
train_encode_file = 'dataset3/train.enc'
train_decode_file = 'dataset3/train.dec'
test_encode_file = 'dataset3/test.enc'
test_decode_file = 'dataset3/test.dec'

UNK_ID = 3


train_encode_vocabulary_file = 'dataset3/train_encode_vocabulary'
train_decode_vocabulary_file = 'dataset3/train_decode_vocabulary'

print("对话转向量...")


# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
    tmp_vocab = []
    with open(vocabulary_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip().decode('utf-8') for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    # {'硕': 3142, 'v': 577, 'Ｉ': 4789, '\ue796': 4515, '拖': 1333, '疤': 2201 ...}
    output_f = open(output_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            line_vec = []
            for words in line.strip().decode('utf-8'):
                line_vec.append(vocab.get(words, UNK_ID))
            output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
    output_f.close()


convert_to_vector(train_encode_file, train_encode_vocabulary_file, 'dataset3/train_encode.vec')
convert_to_vector(train_decode_file, train_decode_vocabulary_file, 'dataset3/train_decode.vec')

convert_to_vector(test_encode_file, train_encode_vocabulary_file, 'dataset3/test_encode.vec')
convert_to_vector(test_decode_file, train_decode_vocabulary_file, 'dataset3/test_decode.vec')