#!/bin/bash

gcloud ai-platform jobs submit training traincloudgpu_tfhub_resnet_gpu_run11 \
--staging-bucket=gs://ai-tpu-experiment \
--package-path=tfhub_resnet_fv_on_gpu \
--module-name=tfhub_resnet_fv_on_gpu.trainer_hub_gpu \
--runtime-version=2.2 \
--python-version=3.7 \
--scale-tier=BASIC_GPU \
--region=us-central1 \
-- \
--distribution_strategy=gpu \
--model_dir=gs://ai-tpu-experiment/traincloudgpu_tfhub_resnet_gpu_run11 \
--train_epochs=3 \
--data_dir=gs://ai-tpu-experiment/tfrecord-flowers \
--num_gpus=4 \
--cache_dir=gs://ai-tpu-experiment/model-cache-dir/imagenet_resnet_v2_50_feature_vector_4