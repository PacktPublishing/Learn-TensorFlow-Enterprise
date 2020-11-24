## Chapter 5 Training at scale

In this chapter, you will learn how to submit a model training job to Google Cloud AI-Platform from your own environment. Basically, you are submitting a training script to the AI-Platform. Training data is stored in your Google Cloud storage bucket, and a staging and destination directories are also Google Cloud storage bucket. 

[tfrecord-dataset](https://github.com/PacktPublishing/learn-tensorflow-enterprise/tree/master/chapter_05/tfrecord_dataset) folder contains the data for you to up;oad to your own Google cloud storage.

The following folders contain topics covered in this chapter:

1. cnn_on_tpu: Custom build a classification model and submit a training script to cloud TPU with distributed training strategy.
2. tfhub_on_tpu: Build a model using TensorFlow Hub's ResNet feature vector and submit a training script to cloud TPU with distributed training strategy.
3. tfhub_on_gpu: Build a model using TensorFlow Hub's ResNet feature vector and submit a training script to cloud GPU with distributed training strategy.

See each folder for instructions.

