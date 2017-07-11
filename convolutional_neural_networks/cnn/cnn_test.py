# -*- coding: utf-8 -*-

from grpc.beta import implementations
# import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import cv2

# from tensorflow_serving.apis.predict_pb2 import PredictResponse

tf.app.flags.DEFINE_string('server', 'localhost:9000','PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Generate test data
img = cv2.imread("../dataset/9.2694.jpg", 0)
rows, cols = img.shape
imagePixes = rows * cols
img_arrayy = img.reshape(1, imagePixes)


# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'test'
request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(img_arrayy, dtype=tf.float32, shape=[1, imagePixes]))
request.inputs['keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))
result = stub.Predict(request, 10.0)  # 10 secs timeout
# print(type(result))

resList = str(result).split('float_val: ')
print('图片数字为0的概率是：'+resList[1].split('\n    ')[0])
print('图片数字为1的概率是：'+resList[2].split('\n    ')[0])
print('图片数字为2的概率是：'+resList[3].split('\n    ')[0])
print('图片数字为3的概率是：'+resList[4].split('\n    ')[0])
print('图片数字为4的概率是：'+resList[5].split('\n    ')[0])
print('图片数字为5的概率是：'+resList[6].split('\n    ')[0])
print('图片数字为6的概率是：'+resList[7].split('\n    ')[0])
print('图片数字为7的概率是：'+resList[8].split('\n    ')[0])
print('图片数字为8的概率是：'+resList[9].split('\n    ')[0])
print('图片数字为9的概率是：'+resList[10].split('\n  ')[0])
