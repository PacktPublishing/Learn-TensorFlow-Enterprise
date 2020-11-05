#!/bin/bash

gcloud ai-platform jobs submit training gcloud_train_run6 \
--staging-bucket=gs://ai-tpu-experiment \
--package-path=custom_model_on_tpu \
--module-name=custom_model_on_tpu.trainer \
--runtime-version=2.2 \
--python-version=3.7 \
--scale-tier=BASIC_TPU \
--region=us-central1 \
-- \
--distribution_strategy=tpu \
--model_dir=gs://ai-tpu-experiment/gcloud_train_run6 \
--train_epochs=3 \
--data_dir=gs://tfrecord-dataset/flowers