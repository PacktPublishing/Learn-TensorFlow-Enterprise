# learn-tensorflow-enterprise

accepted repo invitation

Chapter 6 
local testing:

Replace <FILEPATH> with that of your own environment where this script is saved.

python3 <FILEPATH>/hp_kt_resnet_local_pub.py \
--model_dir=<FILEPATH>/resnet_local_hb_1  \
--cached_basemodel_dir=<FILEPATH>/imagenet_resnet_v2_50_feature_vector_4 \
--train_epoch_best=3 \
--tuner_type=hypoerband
