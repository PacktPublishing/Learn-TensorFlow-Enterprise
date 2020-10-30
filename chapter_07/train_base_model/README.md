## Train a baseline (full) model

We are going to train a flower classifier model. This model is built with TensorFlow Hub's ResNet feature vector. The model will be trained to classify five different types of flowers. 

To launch the training process, make sure you are in this directory. 

```console
python3 default_trainer.py \
--distribution_strategy=default \
--fine_tuning_choice=False \
--train_batch_size=32 \
--validation_batch_size=40 \
--train_epochs=5 \
--data_dir=tf_datasets/flower_photos \
--model_dir=trained_resnet_vector
```
