import kerastuner as kt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os
import IPython
from kerastuner import HyperParameters


from absl import flags
from absl import logging
from absl import app

FLAGS = flags.FLAGS
# flag name, default value, explanation/help.
tf.compat.v1.flags.DEFINE_string('model_dir', 'default_model_dir', 'Directory or bucket for storing checkpoint model.')
tf.compat.v1.flags.DEFINE_bool('fine_tuning_choice', False, 'Retrain base parameters')
tf.compat.v1.flags.DEFINE_integer('train_batch_size', 32, 'Number of samples in a training batch')
tf.compat.v1.flags.DEFINE_integer('validation_batch_size', 40, 'Number of samples in a validation batch')
tf.compat.v1.flags.DEFINE_string('distribution_strategy', 'tpu', 'Distribution strategy for training.')
tf.compat.v1.flags.DEFINE_integer('train_epochs', 3, 'Number of epochs for training')
tf.compat.v1.flags.DEFINE_string('data_dir', 'gs://image-flowers/flower_photos/train', 'Location of training data')
tf.compat.v1.flags.DEFINE_string('cache_dir', 'cached_dir_path' , 'Location of cached model')
tf.compat.v1.flags.DEFINE_integer('num_gpus', 4, 'Numer of GPU per worker')

def model_default():
    flags_obj = flags.FLAGS
    os.environ["TFHUB_CACHE_DIR"] = flags_obj.cache_dir

    IMAGE_SIZE = (224, 224)
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)), 
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 64, activation = 'relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(5, activation='softmax', name = 'custom_class')
    ])

    model.build([None, 224, 224, 3])
 
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.5), 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])
  
    return model

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

def normalize(image, label):
    #Convert `image` from [0, 255] -> [0, 1.0] floats 
    image = tf.cast(image, tf.float32) / 255. + 0.5
    return image, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    flags_obj = flags.FLAGS
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(flags_obj.train_batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def main(_):
    flags_obj = flags.FLAGS
   
    if flags_obj.distribution_strategy == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        strategy_scope = strategy.scope()
        print("All devices: ", tf.config.list_logical_devices('TPU'))
    elif flags_obj.distribution_strategy == 'gpu':
        strategy = tf.distribute.MirroredStrategy()
        strategy_scope = strategy.scope()
        devices = ["device:GPU:%d" % i for i in range(flags_obj.num_gpus)]

    print('NUMBER OF DEVICES: ', strategy.num_replicas_in_sync)

    ## identify data paths and sources
    root_dir = flags_obj.data_dir # this is gs://<bucket>/folder where tfrecord are stored
    file_pattern = "{}/image_classification_builder-train*.tfrecord*".format(root_dir)
    val_file_pattern = "{}/image_classification_builder-validation*.tfrecord*".format(root_dir)

    file_list = tf.io.gfile.glob(file_pattern)
    all_files = tf.data.Dataset.list_files( tf.io.gfile.glob(file_pattern))

    val_file_list = tf.io.gfile.glob(val_file_pattern)
    val_all_files = tf.data.Dataset.list_files( tf.io.gfile.glob(val_file_pattern))

    train_all_ds = tf.data.TFRecordDataset(all_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    val_all_ds = tf.data.TFRecordDataset(val_all_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    # perform data engineering 
    dataset = train_all_ds.map(decode_and_resize)
    val_dataset = val_all_ds.map(decode_and_resize)

    # Create dataset for training run
    BATCH_SIZE = flags_obj.train_batch_size
    VALIDATION_BATCH_SIZE = flags_obj.validation_batch_size
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_ds = val_dataset.batch(VALIDATION_BATCH_SIZE)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = prepare_for_training(dataset)

    FINE_TUNING_CHOICE = True
    NUM_CLASSES = 5
    IMAGE_SIZE = (224, 224)
    
    train_sample_size=0
    for raw_record in train_all_ds:
        train_sample_size += 1
    print('TRAIN_SAMPLE_SIZE = ', train_sample_size)
    validation_sample_size=0
    for raw_record in val_all_ds:
        validation_sample_size += 1
    print('VALIDATION_SAMPLE_SIZE = ', validation_sample_size)

    STEPS_PER_EPOCHS = train_sample_size // BATCH_SIZE
    VALIDATION_STEPS = validation_sample_size // VALIDATION_BATCH_SIZE
    
    model = model_default()
    checkpoint_prefix = os.path.join(flags_obj.model_dir, "train_ckpt_{epoch}")
    callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(flags_obj.model_dir, 'tensorboard_logs')),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
        save_weights_only=True)]

    model.fit(
        train_ds,
        epochs=3, steps_per_epoch=STEPS_PER_EPOCHS,
        validation_data=val_ds,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks)
    
    logging.info('INSIDE MAIN FUNCTION user input model_dir %s', flags_obj.model_dir)
    # Save model trained with chosen HP in user specified bucket location
    model_save_dir = os.path.join(flags_obj.model_dir, 'save_model')
    model.save(model_save_dir)

if __name__ == '__main__':
    app.run(main)