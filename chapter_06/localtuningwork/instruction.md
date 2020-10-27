## Hyperparameter tuning 

To execute the code, once you CD to this directory, run the following command in this directory.

```console
sh tuner_driver.sh
```

This shell script contains the following code:

```console
python3 hp_kt_resnet_local_pub.py \
--model_dir=resnet_local_hb_output  \
--cached_basemodel_dir=imagenet_resnet_v2_50_feature_vector_4 \
--train_epoch_best=2 \
--tuner_type=hyperband
```

This code will execute `hp_kt_resnet_local_pub.py` with the following user input parameters:

`model_dir`: directory where the trained model will be saved. This is going to be under the current directory. 
`cached_basemodel_dir`: directory where the ResNet feature model is stored. This is under the current directory.
`train_epoch_best`: number of epochs to train the model with hyperparameter tuning.
`tuner_type`: The algorithm to use for searching through hyperparameter spaces. Available choices are: `hyperband`, `randomsearch`, and `bayesianoptimization`.