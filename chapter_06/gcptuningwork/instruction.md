## Hyperparameter tuning with cloud TPU

To execute the code in this directory, run the following command in **gcptuningwork** directory.

```console
sh gcp_tuner_driver.sh
```

This shell script contains the following code:

```console
gcloud ai-platform jobs submit training hp_kt_resnet_tpu_hb_test \
--staging-bucket=gs://ai-tpu-experiment \
--package-path=tfk \
--module-name=tfk.tuner.hp_kt_resnet_tpu_act \
--runtime-version=2.2 \
--python-version=3.7 \
--scale-tier=BASIC_TPU \
--region=us-central1 \
--use-chief-in-tf-config="true" \
-- \
--distribution_strategy=tpu \
--data_dir=gs://ai-tpu-experiment/tfrecord-flowers \
--model_dir=gs://ai-tpu-experiment/hp_kt_resnet_tpu_hb_test \
--tuner_type=hyperband
```
This code will execute `hp_kt_resnet_tpu_act.py`, which is in `/tfk/tuner` directory.

Since this uses cloud TPU, the following parameters in the input have to be renamed to your own resources. 

`staging-bucket`: <YOUR_GCLOUD_STORAGE> This is your cloud storage bucket where you stage the training and dependency packages.
`data_dir`: <YOUR_GCLOUD_STORAGE_FOR_TFRECORD> This is your cloud storage bucket where `TFRecord` files are stored.
`model_dir`: <YOUR_GCLOUD_STORAGE_FOLDER> This represents the directory that saves the trained model once the hyperparameter search and training is done.
`tuner_type`: hyperparameter space search algorithm. Choices are `hyperband`, `randomsearch`, `bayesianoptimization`.


### Important notes
In order to make this work, you need `setup.py` placed in **gcptuningwork** directory. This file contains instructions and libraries that points to all the required dependencies (packages, libraries).