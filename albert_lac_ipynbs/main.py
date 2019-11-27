
# coding: utf-8

# In[1]:

HOST_PORT_HTTP = "your-host-port" # TODO: e.g. 123.123.123.123:8511
MODEL_NAME = "albert_remy_lac" # TODO: your model name
VOCAB_FILE = '/data1/albert/zh/albert_base_zh/vocab.txt' # TODO: set vocab.txt path
LABEL_CLASS_FILE = './label_class.txt' # TODO: your label_class.txt
MAX_SEQ_LENGTH = 128


# In[2]:

## ***********************************
## tensorflow-serving-api REST
## ***********************************
import requests
import json
import numpy as np

def predict(input_ids, input_mask, segment_ids, label_ids, hostport_http, version="1"):
    data = json.dumps({
            "signature_name": "serving_default", 
            "instances": [{
                "input_ids": input_ids,
                "input_mask": input_mask,
                "label_ids": label_ids,
                "segment_ids": segment_ids  
            }],
        })
    headers = {"content-type": "application/json"}
    url = 'http://{}/v1/models/{}/versions/{}:predict'.format(hostport_http, MODEL_NAME, version)
    #print(url)
    json_response = requests.post(url, data=data, headers=headers)
    #print(json_response.text)
    predictions = json.loads(json_response.text)['predictions']
    return [np.argmax(x) for pred in predictions for x in pred ]


# In[3]:

class Example(object):
    def __init__(self, text_a, label):
        self.text_a = text_a
        self.label = label
    
def convert_single_example(example, label_map, max_seq_length, tokenizer):
    textlist = list(example.text_a)
    labellist = example.label.split(' ')
   
    if len(textlist) != len(labellist):
        print(textlist, labellist)
        print(len(textlist), len(labellist))

    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    ntokens = []
    nlabels = []
    segment_ids = []
    label_ids = []

    ntokens.append("[CLS]")
    nlabels.append("[CLS]")
    label_ids.append(label_map["[CLS]"]) # append("O") or append("[CLS]") not sure!
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        nlabels.append(labels[i])
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")
    nlabels.append("[SEP]")
    label_ids.append(label_map["[SEP]"])
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0) # we don't concerned about it!
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, label_ids


# In[4]:

import tokenization
import merge_lac

def get_labels(label_class_file):
    with open(label_class_file, 'r', encoding='utf8') as r:
        return [x.strip() for x in r.readlines()]
    
# 获取tokenizer
do_lower_case = True
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=do_lower_case)

# 构造 label_id_map 和 id_label_map
label_list = get_labels(LABEL_CLASS_FILE)
label_id_map = {}
id_label_map = {}
for (i, label) in enumerate(label_list, 1):
    label_id_map[label] = i
    id_label_map[i] = label


# In[6]:

texts = [
    '查一下餐包有哪些做法',
    '瘦肉末或者猪肉丸能做什么',
    '猪肉丁可以和什么一起煮汤',
    '看看蒜香排骨的做法',
    '如何调理不正常的日常膳食',
    '能介绍些养颜美容的食物吗',
    '健康饮食：怎样吃海鲜更安全',
    '吃大闸蟹应该注意什么',
    ]

# 预测结果 [CLS] ...
def predict_with_version(version):
    for text in texts:
        example = Example(text, ' '.join(['O'] * len(text)))
        # 转换为模型需要的example
        input_ids, input_mask, segment_ids, label_ids = convert_single_example(example, label_id_map, MAX_SEQ_LENGTH, tokenizer)
        #print(input_ids)

        predict_label_ids = predict(input_ids, input_mask, segment_ids, label_ids, HOST_PORT_HTTP, version)

        # predict_label_ids[1:] 去掉 CLS 的词性
        predict_labels = [id_label_map[labelid] for labelid in predict_label_ids[1:] if labelid != 0]

        print(merge_lac.merge_line2(list(text), predict_labels))

predict_with_version('1574843457') # TODO: repalce the version with yours


# In[ ]:



