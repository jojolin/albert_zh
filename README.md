## albert lac
- 该项目使用[albert_zh](https://github.com/brightmart/albert_zh)的预训练模型进行lac标注的finetune. 并通过导出为SavedModel模型，使用tf.serving (docker)提供服务.
- see also [README.md](./README.ORI.md)

## 添加的主要代码
``` python
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):

  # ......

    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_sum(per_example_loss)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return (loss, per_example_loss, logits, probabilities)


def main(_):

  # ......

  if FLAGS.do_export:    # export model for serving

    def serving_input_receiver_fn():
      input_ids  = tf.placeholder(tf.int32, 
          shape=[None, FLAGS.max_seq_length], name="input_ids")
      input_mask = tf.placeholder(tf.int32, 
          shape=[None, FLAGS.max_seq_length], name="input_mask")
      segment_ids = tf.placeholder(tf.int32, 
          shape=[None, FLAGS.max_seq_length], name="segment_ids")
      label_ids = tf.placeholder(tf.int32, 
          shape=[None, FLAGS.max_seq_length], name="label_ids")
      features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label_ids": label_ids
      }
      receiver_tensors = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label_ids": label_ids
      }
      return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,

    # ......

      predictions={"probabilities": probabilities} 
      output = {'serving_default': tf.estimator.export.PredictOutput(predictions)}
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predictions,
          scaffold_fn=scaffold_fn,
          export_outputs=output)
    return output_spec

```

## train
- 下载预训练模型, 参考[albert_tiny](https://github.com/brightmart/albert_zh) 
- 数据准备
  - 文件结构

```
├── dev.tsv
├── label_class.txt
├── test.tsv
└── train.tsv
```

- \*.tsv
  - 其中, ^B: \x02, 语句和标签用\t隔开
  - lac 标签复用百度[PaddleNLP](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis)

``` 
text_a	label
苦^B瓜^B味^B苦^B，^B性^B寒^B，^B归^B心^B、^B肺^B、^B脾^B、^B胃^B经^B，^B具^B有^B消^B暑^B清^B热^B，^B解^B毒^B健^B胃^B，^B除^B邪^B热^B，^B聪^B耳^B明^B目^B，^B润^B泽^B肌^B肤^B，^B强^B身^B，^B使^B人^B精^B力^B旺^B盛^B，^B不^B易^B衰^B老^B的^B功^B效^B，^B还^B有^B降^B血^B糖^B、^B抗^B肿^B瘤^B、^B抗^B病^B毒^B、^B抗^B菌^B、^B促^B进^B免^B疫^B力^B等^B作^B用^B。  n-B^Bn-I^Ba-B^Ba-I^Bw-B^Bn-B^Ba-B^Bw-B^Bv-B^Bv-I^Bw-B^Bn-B^Bw-B^Bn-B^Bw-B^Bn-B^Bn-I^Bw-B^Bv-B^Bv-I^Bv-B^Bv-I^Bv-B^Bv-I^Bw-B^Bv-B^Bv-I^Bv-B^Bv-I^Bw-B^Bp-B^Ban-B^Ban-I^Bw-B^Bv-B^Bv-I^Bv-I^Bv-I^Bw-B^Bv-B^Bv-I^Bn-B^Bn-I^Bw-B^Bv-B^Bv-I^Bw-B^Bn-B^Bn-I^Bn-B^Bn-I^Ba-B^Ba-I^Bw-B^Bad-B^Bad-I^Bv-B^Bv-I^Bu-B^Bn-B^Bn-I^Bw-B^Bv-B^Bv-I^Bvn-B^Bvn-I^Bvn-I^Bw-B^Bv-B^Bn-B^Bn-I^Bw-B^Bnz-B^Bnz-I^Bnz-I^Bw-B^Bvn-B^Bvn-I^Bw-B^Bv-B^Bv-I^Bn-B^Bn-I^Bn-I^Bu-B^Bn-B^Bn-I^Bw-B
黄^B豆^B芽^B富^B含^B蛋^B白^B质^B、^B维^B生^B素^B、^B粗^B纤^B维^B、^B胡^B萝^B卜^B素^B、^B钙^B、^B磷^B、^B铁^B等^B营^B养^B元^B素^B。^B其^B所^B含^B的^B维^B生^B素^Bc^B能^B营^B养^B毛^B
发^B，^B使^B头^B发^B保^B持^B乌^B黑^B光^B亮^B，^B对^B面^B部^B雀^B斑^B有^B较^B好^B的^B淡^B化^B作^B用^B。^B其^B所^B含^B的^B维^B生^B素^Be^B能^B保^B护^B皮^B肤^B和^B毛^B细^B血^B管^B，^B
防^B止^B动^B脉^B硬^B化^B，^B防^B治^B老^B年^B高^B血^B压^B。        nz-B^Bnz-I^Bnz-I^Bv-B^Bv-I^Bnz-B^Bnz-I^Bnz-I^Bw-B^Bn-B^Bn-I^Bn-I^Bw-B^Ba-B^Bn-B^Bn-I^Bw-B^Bnz-B^Bnz-I^Bnz-I^Bnz-I
^Bw-B^Bn-B^Bw-B^Bn-B^Bw-B^Bn-B^Bu-B^Bn-B^Bn-I^Bn-B^Bn-I^Bw-B^Br-B^Bu-B^Bv-B^Bu-B^Bnz-B^Bnz-I^Bnz-I^Bnz-I^Bv-B^Bn-B^Bn-I^Bn-B^Bn-I^Bw-B^Bv-B^Bn-B^Bn-I^Bv-B^Bv-I^Ba-B^Ba-I^Ba-B^Ba-I
^Bw-B^Bp-B^Bn-B^Bn-I^Bn-B^Bn-I^Bv-B^Ba-B^Ba-I^Bu-B^Bvn-B^Bvn-I^Bn-B^Bn-I^Bw-B^Br-B^Bu-B^Bv-B^Bu-B^Bn-B^Bn-I^Bn-I^Bxc-B^Bv-B^Bv-B^Bv-I^Bn-B^Bn-I^Bc-B^Bn-B^Bn-I^Bn-I^Bn-I^Bw-B^Bv-B^Bv-I^Bvn-B^Bvn-I^Bvn-I^Bvn-I^Bw-B^Bv-B^Bv-I^Bn-B^Bn-I^Bn-B^Bn-I^Bn-I^Bw-B
``` 
- label\_class.txt

```
a-B
a-I
ad-B
ad-I
an-B
an-I
c-B
c-I
d-B
...
```

- 执行finetune: `sh run_lac_remy.sh train`

- 执行predict: `sh run_lac_remy.sh test`

## Serving
- 导出模型: `sh run_lac_remy.sh export`
  - 查看目录`cd export_serving_remy_lac`

```
export_serving_remy_lac/
├── 1574843457
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── models.config
```
  - 其中,`models.config`为自己添加的文件, `1574843457`为导出模型版本号

```
model_config_list {
  config {
    name: 'albert_remy_lac'
    base_path: '/models/albert_remy_lac'
    model_platform: 'tensorflow'
    model_version_policy {
        specific {
            versions: 1574843457
        }
    }
  }
}

```

- 利用`docker tf.serving`提供服务
  - `cd albert_lac_ipynbs`
  - 创建软链：`ln -s ../export_serving_remy_lac albert_remy_lac`
  - 运行tfserving, no gpu: `sh start-tf1.14.0-rc.sh albert_remy_lac`
  - 运行tfserving, gpu: `sh start-tfgpu1.14.0.sh albert_remy_lac`

```
albert_lac_ipynbs/
├── label_class.txt
├── main.ipynb
├── main.py
├── merge_lac.py
├── __pycache__
│   ├── merge_lac.cpython-36.pyc
│   └── tokenization.cpython-36.pyc
├── start-tf1.14.0-rc.sh
├── start-tfgpu1.14.0.sh
└── tokenization.py

``` 

## 提供服务
- 查看模型接口标签 `saved_model_cli show --dir '$dir1' --tag_set serve --signature_def serving_default`
  - $dir1: 'export\_serving\_remy\_lac/1574843457'

```
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, 128)
      name: input_ids:0
  inputs['input_mask'] tensor_info:
      dtype: DT_INT32
      shape: (-1, 128)
      name: input_mask:0
  inputs['label_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, 128)
      name: label_ids:0
  inputs['segment_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, 128)
      name: segment_ids:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['probabilities'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 128, 59)
      name: loss/Softmax:0
Method name is: tensorflow/serving/predict
```

- 构造访问rest api. 参考`albert_lac_ipynbs/main.ipynb` 或者 `albert_lac_ipynbs/main.py`
``` python
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
``` 

- 模型预测结果tiny:

```
['查/v', '一下/m', '餐/nz', '包/n', '有/v', '哪些/r', '做法/n']
['瘦肉末/nz', '或者/c', '猪肉丸/nz', '能/v', '做/v', '什么/r']
['猪肉/n', '丁/nz', '可以/v', '和/p', '什么/r', '一起/d', '煮汤/vn']
['看看/v', '蒜/n', '香/a', '排骨/v', '的/u', '做法/n']
['如何/d', '调理/v', '不/d', '正常/a', '的/u', '日常/a', '膳食/n']
['能/v', '介绍/v', '些/q', '养颜/vn', '美容/vn', '的/u', '食物/n', '吗/xc']
['健康/a', '饮食：/n', '怎样/d', '吃/v', '海鲜/n', '更/d', '安全/a']
['吃/v', '大闸蟹/nz', '应该/v', '注意/v', '什么/r']
```
- 预测性能
  - gpu: GeForce RTX 2080-ti
  - [tiny版](https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip): 单线程，100个请求, ~3s/~2s (no gpu/gpu)
  - [base版](https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip): 单线程，100个请求, ~10s/~5s(no gpu/gpu)

- 准确性

```
数据量

    2001 dev.tsv
    1190 test.tsv
   10000 train.tsv

tiny版:
Eval results albert_tiny_remy_lac_checkpoints/model.ckpt-18748 
eval_accuracy = 0.95755637
eval_loss = 0.32848656
global_step = 18748
loss = 335.19763

base版

```

## Thanks
- [albert_zh](https://github.com/brightmart/albert_zh)
- [PaddleNLP](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis)
