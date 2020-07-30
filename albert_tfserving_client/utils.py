from . import tokenization
from .model_config import VOCAB_FILE, LABEL_CLASS_FILE


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
    label_ids.append(label_map["[CLS]"])  # append("O") or append("[CLS]") not sure!
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
    # label_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)  # we don't concerned about it!
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, label_ids


def get_labels(label_class_file):
    with open(label_class_file, 'r', encoding='utf8') as r:
        return [x.strip() for x in r.readlines()]


# 获取tokenizer
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

# 构造 label_id_map 和 id_label_map
label_list = get_labels(LABEL_CLASS_FILE)
label_id_map = {}
id_label_map = {}
for (i, label) in enumerate(label_list, 1):
    label_id_map[label] = i
    id_label_map[i] = label
