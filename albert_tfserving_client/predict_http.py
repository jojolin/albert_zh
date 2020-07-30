## ***********************************
## tensorflow-serving-api REST
## ***********************************

import json

import numpy as np
import requests

from . import merge_lac
from . import utils
from .model_config import MAX_SEQ_LENGTH
from .model_config import MODEL_NAME
from .utils import Example
from .utils import convert_single_example


class PredictHTTP(object):
    def __init__(self, hostport, version):
        self.hostport = hostport
        self.predict_url = 'http://{}/v1/models/{}/versions/{}:predict'.format(hostport, MODEL_NAME, version)
        self.headers = {"content-type": "application/json"}

    def predict_http(self, converted_example):
        input_ids, input_mask, segment_ids, label_ids = converted_example
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": [{
                "input_ids": input_ids,
                "input_mask": input_mask,
                "label_ids": label_ids,
                "segment_ids": segment_ids
            }],
        })

        json_response = requests.post(self.predict_url, data=data, headers=self.headers)
        predictions = json.loads(json_response.text)['predictions']
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
        converted_example = convert_single_example(example, utils.label_id_map, MAX_SEQ_LENGTH, utils.tokenizer)
        cls_prob, predict_label_ids = self.predict_http(converted_example)
        predict_labels = [utils.id_label_map[labelid] for labelid in predict_label_ids if labelid != 0]
        return cls_prob, predict_labels[0], merge_lac.merge_line2(list(" " + text), predict_labels)
