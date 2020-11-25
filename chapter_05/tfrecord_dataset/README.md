## Upload `TFRecord` dataset to your cloud storage

There are two things you need to upload to your own cloud storage: 
`TFRecord` dataset for the five types of flowers partitioned for training, test and validation.
Pre-built ResNet feature vector (originally downloded from [`TensorFlow Hub`](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4))
You may upload the content to your cloud storage. 

You should upload [`flower`](https://github.com/PacktPublishing/learn-tensorflow-enterprise/tree/master/chapter_05/tfrecord_dataset/flowers) directory into your cloud storage bucket. You may go to GCP portal to create a bucket and upload the directory.

Once you are done, below is an image of how it should look like:
![](s1.png)

## Upload pre-built ResNet feature vector model to your cloud storage

It is a good idea to cache the pre-built model from TensorFlow Hub beforehand, so it is always accessible in case connectivity issues occur.

You should to upload [`model-cache-dir`](https://github.com/PacktPublishing/learn-tensorflow-enterprise/tree/master/chapter_05/tfrecord_dataset/model-cache-dir) into your cloud storage buchet. You may go to GCP portal to create a bucket and upload the directory.