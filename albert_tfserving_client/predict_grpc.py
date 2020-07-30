import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from . import merge_lac
from . import utils
from .model_config import MAX_SEQ_LENGTH, MODEL_NAME
from .utils import Example


class PredictGRPC(object):
    def __init__(self, hostport, version):
        self.hostport = hostport

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = MODEL_NAME
        self.request.model_spec.signature_name = 'serving_default'
        self.request.model_spec.version.value = version
        # 提前初始化,第一个请求会比较耗时
        # self.request.inputs['input_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(
        #     np.ones(MAX_SEQ_LENGTH), shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
        # self.request.inputs['input_mask'].CopyFrom(tf.contrib.util.make_tensor_proto(
        #     np.ones(MAX_SEQ_LENGTH), shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
        # self.request.inputs['label_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(
        #     np.ones(MAX_SEQ_LENGTH), shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
        # self.request.inputs['segment_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(
        #     np.ones(MAX_SEQ_LENGTH), shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))

    def predict_grpc(self, converted_example):
        input_ids, input_mask, segment_ids, label_ids = converted_example
        self.request.inputs['input_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(
            input_ids, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
        self.request.inputs['input_mask'].CopyFrom(tf.contrib.util.make_tensor_proto(
            input_mask, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
        self.request.inputs['label_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(
            label_ids, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
        self.request.inputs['segment_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(
            segment_ids, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))

        channel = grpc.insecure_channel(self.hostport)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        result = self.stub.Predict(self.request)
        predictions = np.reshape(result.outputs['probabilities'].float_val, (1, MAX_SEQ_LENGTH, -1))
        cls_pred = predictions[0][0]  # 分类概率
        cls_prob = cls_pred[np.argmax(cls_pred)]  # 分类概率值
        return cls_prob, [np.argmax(x) for pred in predictions for x in pred]

    def predict(self, text):
        """
        预测结果 [CLS] ...
        :param text:
        :return:
        """
        example = Example(text, ' '.join(['O'] * len(text)))
        converted_example = utils.convert_single_example(example, utils.label_id_map, MAX_SEQ_LENGTH, utils.tokenizer)
        cls_prob, predict_label_ids = self.predict_grpc(converted_example)
        predict_labels = [utils.id_label_map[labelid] for labelid in predict_label_ids if labelid != 0]
        return cls_prob, predict_labels[0], merge_lac.merge_line2(list(" " + text), predict_labels)
