from batchcreator import DataGenerator, get_list_IDs
from datetime import datetime
from model_builder import GAN, build_generator
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import logger
import numpy as np

print('Starting test run')
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Setup wandb run
run = wandb.init(project='precipitation-forecasting',
            config={
            'batch_size' : 32,
            'epochs': 200,
            'lr_g': 0.001,
            'lr_d': 0.001,
            'l_g': 0.006,
            'l_rec': 1,
            'g_cycles': 3,
            'noise_labels': 0,
            'x_length': 6,
            'y_length': 1,
            'rnn_type': 'GRU',
            'filter_no_rain': 'avg0.01mm',
            'train_data': 'train2015_2018.npy',
            'val_data': 'val2019.npy',
            'architecture': 'AENN',
            'model': 'Generator',
            'norm_method': 'minmax',
            'downscale256': True,
            'convert_to_dbz': True,
            'load_prep': True,
            'server':  'RU'
        })
config = wandb.config

# Create generator for training
list_IDs = np.load(config.train_data, allow_pickle = True)
print('Samples in training set:')
print(len(list_IDs))

generator = DataGenerator(list_IDs, batch_size=config.batch_size,
                          x_seq_size=config.x_length, y_seq_size=config.y_length,
                          norm_method = config.norm_method, load_prep=config.load_prep,
                          downscale256 = config.downscale256, convert_to_dbz = config.convert_to_dbz)

if config.val_data:
    val_IDs = np.load(config.val_data, allow_pickle = True)
    print('Samples in validation set:')
    print(len(val_IDs))

    validation_generator = DataGenerator(val_IDs, batch_size = config.batch_size,
                                     x_seq_size = config.x_length, y_seq_size = config.y_length,
                                     norm_method = config.norm_method, load_prep = config.load_prep,
                                     downscale256 = config.downscale256, convert_to_dbz = config.convert_to_dbz)
else:
    validation_generator = None
# Initialize model
if config.model == 'GAN':
    model = GAN(rnn_type = config.rnn_type, x_length = config.x_length, y_length = config.y_length,
             architecture = config.architecture, g_cycles=config.g_cycles, noise_labels = config.noise_labels,
                l_g = config.l_g, l_rec = config.l_rec, norm_method = config.norm_method, downscale256 = config.downscale256)
    model.compile(lr_g = config.lr_g, lr_d = config.lr_d)
    callbacks = [WandbCallback(), logger.ImageLogger(generator), logger.GradientLogger(generator)]
else:
    model = build_generator(architecture=config.architecture, rnn_type=config.rnn_type, relu_alpha=0.2,
			x_length = config.x_length, y_length = config.y_length, norm_method = config.norm_method, downscale256 = config.downscale256)
    opt = tf.keras.optimizers.Adam(learning_rate=config.lr_g)
    model.compile(loss='mse', metrics=['mse', 'mae'])
    callbacks = [WandbCallback(), logger.ImageLogger(generator)]

history = model.fit(generator, validation_data = validation_generator, epochs = config.epochs,
                    callbacks = callbacks)
