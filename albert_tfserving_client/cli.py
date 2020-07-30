import os
import sys
import time

from .model_config import HOST_PORT_GRPC, HOST_PORT_HTTP, ALBERT_DIR
from .predict_grpc import PredictGRPC
from .predict_http import PredictHTTP

if __name__ == '__main__':
    way = sys.argv[1]

    if way == 'http':
        predicter = PredictHTTP(HOST_PORT_HTTP, version=1595990910)
    elif way == 'grpc':
        predicter = PredictGRPC(HOST_PORT_GRPC, version=1595990910)
    else:
        print('no such way!')
        sys.exit(0)

    cases = [x.strip() for x in
             open(os.path.join(ALBERT_DIR, 'multi.txt'), 'r', encoding='utf8').readlines()]
    st = time.time()
    for x in cases:
        cls_prob, cls_label, labels = predicter.predict(x)
        print(f'{x}: {cls_label}({cls_prob}), {" ".join(labels)}')
    print(time.time() - st)
