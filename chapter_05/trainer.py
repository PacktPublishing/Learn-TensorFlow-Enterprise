import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from absl import flags
from absl import logging
from absl import app

FLAGS = flags.FLAGS

tf.compat.v1.flags.DEFINE_string('model', 'DefaultModel', 'model to run') # name ,default, help
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.compat.v1.flags.DEFINE_string('distribution_strategy', 'tpu', 'Distribution strategy for training.')
tf.compat.v1.flags.DEFINE_integer('height_pixel', 224, 'Number of pixel for image height')
tf.compat.v1.flags.DEFINE_integer('width_pixel', 224, 'Number of pixel for image width')
tf.compat.v1.flags.DEFINE_bool('fine_tuning_choice', True, 'Retrain base parameters')
tf.compat.v1.flags.DEFINE_integer('classes', 5, 'Number of classes for classification')
tf.compat.v1.flags.DEFINE_integer('train_batch_size', 32, 'Number of samples in a training batch')
tf.compat.v1.flags.DEFINE_integer('validation_batch_size', 40, 'Number of samples in a validation batch')
tf.compat.v1.flags.DEFINE_string('model_dir', 'default_model_dir', 'Directory or path for storing checkpoint model.')
tf.compat.v1.flags.DEFINE_integer('train_epochs', 3, 'Number of epochs for training')
tf.compat.v1.flags.DEFINE_string('data_dir', 'gs://image-flowers/flower_photos/train', 'Location of training data')

def run(flags_obj):

    print('TENSORFLOW_VERSION: ', tf.__version__)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    strategy_scope = strategy.scope()
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    ########## Below is ResNet training ##############
    root_dir = flags_obj.data_dir # this is gs://<bucket>/folder where tfrecord are stored
    train_file_pattern = "{}/image_classification_builder-train*.tfrecord*".format(root_dir)
    val_file_pattern = "{}/image_classification_builder-validation*.tfrecord*".format(root_dir)

    train_all_files = tf.data.Dataset.list_files( tf.io.gfile.glob(train_file_pattern))
    val_all_files = tf.data.Dataset.list_files( tf.io.gfile.glob(val_file_pattern))

    train_all_ds = tf.data.TFRecordDataset(train_all_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    val_all_ds = tf.data.TFRecordDataset(val_all_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    def decode_and_resize(serialized_example):
        # resized image should be [224, 224, 3] and normalized to value range [0, 255] 
        # label is integer index of class.
    
        parsed_features = tf.io.parse_single_example(
        serialized_example,
        features = {
        'image/channels' :  tf.io.FixedLenFeature([], tf.int64),
        'image/class/label' :  tf.io.FixedLenFeature([], tf.int64),
        'image/class/text' : tf.io.FixedLenFeature([], tf.string),
        'image/colorspace' : tf.io.FixedLenFeature([], tf.string),
        'image/encoded' : tf.io.FixedLenFeature([], tf.string),
        'image/filename' : tf.io.FixedLenFeature([], tf.string),
        'image/format' : tf.io.FixedLenFeature([], tf.string),
        'image/height' : tf.io.FixedLenFeature([], tf.int64),
        'image/width' : tf.io.FixedLenFeature([], tf.int64)
        })
        image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
        label = tf.cast(parsed_features['image/class/label'], tf.int32)
        label_txt = tf.cast(parsed_features['image/class/text'], tf.string)
        label_one_hot = tf.one_hot(label, depth = 5)
        resized_image = tf.image.resize(image, [224, 224], method='nearest')
        return resized_image, label_one_hot

    train_dataset = train_all_ds.map(decode_and_resize)
    val_dataset = val_all_ds.map(decode_and_resize)
 
    def normalize(image, label):
        #Convert `image` from [0, 255] -> [0, 1.0] floats 
        image = tf.cast(image, tf.float32) / 255. + 0.5
        return image, label

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    with strategy.scope():
        base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax', name = 'custom_class')
        ])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.05, decay_steps=100000, decay_rate=0.96)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                metrics=['accuracy'])

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = flags_obj.train_batch_size
    VALIDATION_BATCH_SIZE = flags_obj.validation_batch_size
    train_dataset = train_dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    val_ds = val_dataset.batch(VALIDATION_BATCH_SIZE)   
    train_ds = prepare_for_training(train_dataset)

    checkpoint_prefix = os.path.join(flags_obj.model_dir, "ckpt_{epoch}")

    callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(flags_obj.model_dir, 'tensorboard_logs')),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)]

    train_sample_size=0
    for raw_record in train_all_ds:
        train_sample_size += 1
    print('TRAIN_SAMPLE_SIZE = ', train_sample_size)
    validation_sample_size=0
    for raw_record in val_all_ds:
        validation_sample_size += 1
    print('VALIDATION_SAMPLE_SIZE = ', validation_sample_size)

    steps_per_epoch = train_sample_size // BATCH_SIZE
    validation_steps = validation_sample_size // VALIDATION_BATCH_SIZE

    hist = model.fit(
        train_ds,
        epochs=flags_obj.train_epochs, steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks)

    model_save_dir = os.path.join(flags_obj.model_dir, 'save_model')
    model.save(model_save_dir)

def main(_):
   logging.info('LOGGING INFO selected model %s', FLAGS.model)
   logging.info('LOGGING INFO train_epochs %d', FLAGS.train_epochs)
   logging.info('LOGGING INFO distribution_strategy %s', FLAGS.distribution_strategy)

   # Now pass all user input parameters as flags into run function
   run(flags.FLAGS)
   
if __name__ == '__main__':
   logging.set_verbosity(logging.INFO)
   app.run(main)