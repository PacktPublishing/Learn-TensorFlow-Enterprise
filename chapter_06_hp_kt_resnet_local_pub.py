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

# flag name, default value, explanation/help.
tf.compat.v1.flags.DEFINE_string('model_dir', 'default_model_dir', 'Directory or bucket for storing checkpoint model.')
tf.compat.v1.flags.DEFINE_bool('fine_tuning_choice', False, 'Retrain base parameters')
tf.compat.v1.flags.DEFINE_integer('train_batch_size', 32, 'Number of samples in a training batch')
tf.compat.v1.flags.DEFINE_integer('validation_batch_size', 40, 'Number of samples in a validation batch')
tf.compat.v1.flags.DEFINE_string('cached_basemodel_dir', 'default_dir', 'Cached base model')
tf.compat.v1.flags.DEFINE_string('tuner_type', 'Hyperband', 'Type of tuner; default is hyperband')
tf.compat.v1.flags.DEFINE_integer('train_epoch_best', 3, 'Epoch for training with best hyperparameters')

def get_builtin_data():
    """ 
    Return an object representing the input data directory
    """
    data_dir = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
    return data_dir

def make_generators(data_dir, flags_obj):
    """
    Build image generators for training and validation images.
    Image data are resampled to ResNet dimensions and values normalized
    """
    BATCH_SIZE = flags_obj.train_batch_size
    IMAGE_SIZE = (224, 224)
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    do_data_augmentation = False 
    if do_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            **datagen_kwargs)
    else:
        train_datagen = valid_datagen
        train_generator = train_datagen.flow_from_directory(
            data_dir, subset="training", shuffle=True, **dataflow_kwargs)

    return train_generator, valid_generator

def map_labels(train_generator):
    """
    Returns a dictionary that maps index back to label.
    Useful for scoring and interpreting predictions.
    """
    labels_idx = (train_generator.class_indices)
    idx_labels = dict((v,k) for k,v in labels_idx.items())
    return idx_labels


def model_builder(hp):
    """
    Build the model structure with hyperparameters in place.
    Provide cached base model in case http request cannot be fulfilled.
    """
    flags_obj = flags.FLAGS
    os.environ["TFHUB_CACHE_DIR"] = flags_obj.cached_basemodel_dir
    hp_units = hp.Int('units', min_value = 64, max_value = 256, step = 64)
    IMAGE_SIZE = (224, 224)
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)), 
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = hp_units, activation = 'relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(5, activation='softmax', name = 'custom_class')
    ])

    model.build([None, 224, 224, 3])
    
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=hp_learning_rate, momentum=0.5), 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])
  
    return model

def main(_):
    flags_obj = flags.FLAGS

    strategy = tf.distribute.MirroredStrategy()

    data_dir = get_builtin_data()
    train_gtr, validation_gtr = make_generators(data_dir, flags_obj)
    idx_labels = map_labels(train_gtr)
     
    """Runs the hyperparameter search."""
    if(flags_obj.tuner_type.lower() == 'BayesianOptimization'.lower()):
        tuner = kt.tuners.BayesianOptimization(
            hypermodel = model_builder,
            objective ='val_accuracy',
            tune_new_entries = True,
            allow_new_entries = True,
            max_trials = 5,
            directory = flags_obj.model_dir,
            project_name = 'hp_tune_bo',
            overwrite = True
            )
    elif (flags_obj.tuner_type.lower() == 'RandomSearch'.lower()):
        tuner = kt.RandomSearch(
            hypermodel = model_builder, 
            objective='val_accuracy',
            tune_new_entries = True, 
            allow_new_entries = True,
            max_trials = 5,
            directory = flags_obj.model_dir,
            project_name = 'hp_tune_rs',
            overwrite = True)
    else:
        tuner = kt.Hyperband(
            hypermodel = model_builder,
            objective = 'val_accuracy', 
            max_epochs = 3,
            factor = 2,
            distribution_strategy=strategy,
            directory = flags_obj.model_dir,
            project_name = 'hp_tune_hb',
            overwrite = True)
            
    tuner.search(train_gtr,
        steps_per_epoch=train_gtr.samples // train_gtr.batch_size,
        validation_data=validation_gtr,
        validation_steps=validation_gtr.samples // validation_gtr.batch_size,
        epochs=3,
        callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy')])


    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    print(f"""
        The hyperparameter search is done. 
        The best number of nodes in the dense layer is {best_hps.get('units')}.
        The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        """)

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    checkpoint_prefix = os.path.join(flags_obj.model_dir, "best_hp_train_ckpt_{epoch}")
    callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(flags_obj.model_dir, 'tensorboard_logs')),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)]

    steps_per_epoch = train_gtr.samples // train_gtr.batch_size
    validation_steps = validation_gtr.samples // validation_gtr.batch_size
    model.fit(
        train_gtr,
        epochs=flags_obj.train_epoch_best, steps_per_epoch=steps_per_epoch,
        validation_data=validation_gtr,
        validation_steps=validation_steps,
        callbacks=callbacks)
    
    logging.info('INSIDE MAIN FUNCTION user input model_dir %s', flags_obj.model_dir)
    # Save model trained with chosen HP in user specified bucket location
    model_save_dir = os.path.join(flags_obj.model_dir, 'best_save_model')
    model.save(model_save_dir)



if __name__ == '__main__':
    app.run(main)