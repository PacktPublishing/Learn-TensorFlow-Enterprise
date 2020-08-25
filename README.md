# learn-tensorflow-enterprise

accepted repo invitation

Chapter 6:

Below is an example command to run the script at local node for testing and learning.
Replace FILEPATH with that of your own environment where this script is saved.
  
Download ResNet feature vector from TensorFlow hub and extract it in cached_basemodel_dir of your local directory.
https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4


python3 FILEPATH/chapter_06_hp_kt_resnet_local_pub.py \
--model_dir=FILEPATH/resnet_local_hb_1  \
--cached_basemodel_dir=FILEPATH/imagenet_resnet_v2_50_feature_vector_4 \
--train_epoch_best=3 \
--tuner_type=hypoerband
