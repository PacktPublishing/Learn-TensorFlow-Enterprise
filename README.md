# learn-tensorflow-enterprise

accepted repo invitation

# Chapter 6

Below is an example command to run the script at local node for testing and learning.
Replace FILEPATH with that of your own environment where this script is saved.
  
Download ResNet feature vector from TensorFlow hub and extract it in cached_basemodel_dir of your local directory.
https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4

Run this in your local environment's command terminal, it is executed by local compute:

python3 FILEPATH/chapter_06_hp_kt_resnet_local_pub.py \
--model_dir=FILEPATH/resnet_local_hb_1  \
--cached_basemodel_dir=FILEPATH/imagenet_resnet_v2_50_feature_vector_4 \
--train_epoch_best=3 \
--tuner_type=hypoerband

Below is an example command to run the script at your local node, and submit training job to Google Cloud AI Platform using TPU.

Download Resnet feature vector from TensorFlow hub and place it in your own Gooogle Cloud storage bucket instead of local directory.

Note: in the example command below, the command is in a directory where setup.py is stored. in this directory, there is a 'python' directory, which has 'ScriptProject' directory, which contains hp_kt_resnet_tpu_act.py.

Run this in your local environment's command terminal, it will be executed in GCP AI-Platform:

gcloud ai-platform jobs submit training hp_kt_resnet_tpu_hb_config \
--staging-bucket=gs://[YOUR_BUCKET_NAME] \
--package-path=python \
--module-name=python.ScriptProject.hp_kt_resnet_tpu_act \
--runtime-version=2.1 \
--python-version=3.7 \
--scale-tier=BASIC_TPU \
--region=us-central1 \
--use-chief-in-tf-config="true" \
-- \
--distribution_strategy=tpu \
--data_dir=gs://[BUCKET_NAME]/tfrecord-flowers \
--model_dir=gs://[BUCKET_NAME]/hp_kt_resnet_tpu_hb_config \
--tuner_type=HYPERBAND
