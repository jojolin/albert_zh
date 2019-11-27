#!/bin/sh

MODELNAME=$1
MODEL_DIR="$(pwd)"

# Start TensorFlow Serving container and open GRPC/REST API port
docker run --name "$MODELNAME"_114rc0 -t --rm -p 8511:8501 -p 8510:8500 \
       -v "$MODEL_DIR/$MODELNAME:/models/$MODELNAME" \
       -v "$MODEL_DIR/$MODELNAME/models.config:/models/models.config" \
       tensorflow/serving:1.14.0-rc0 \
       --model_config_file=/models/models.config
