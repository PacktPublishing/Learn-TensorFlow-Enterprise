#!/bin/bash

gcloud ai-platform jobs submit training gcloud_train_tfhub_run10 \
--staging-bucket=gs://ai-tpu-experiment \
--package-path=tfhub_resnet_fv_on_tpu \
--module-name=tfhub_resnet_fv_on_tpu.trainer_hub \
--runtime-version=2.2 \
--python-version=3.7 \
--scale-tier=BASIC_TPU \
--region=us-central1 \
-- \
--distribution_strategy=tpu \
--model_dir=gs://ai-tpu-experiment/gcloud_train_tfhub_run10 \
--train_epochs=10 \
--data_dir=gs://ai-tpu-experiment/tfrecord-flowers \
--cache_dir=gs://ai-tpu-experiment/model-cache-dir/imagenet_resnet_v2_50_feature_vector_4