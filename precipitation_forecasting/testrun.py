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
            'batch_size' : 8,
            'epochs': 100,
            'lr_g': 0.001,
            'lr_d': None,
            'x_length': 6,
            'y_length': 1,
            'rnn_type': 'GRU',
            'filter_no_rain': True,
            'dataset': '2019-06-06/08',
            'architecture': 'AENN',
            'model': 'Generator Only',
            'norm_method': None
        })
config = wandb.config

# Create generator for training
start_dt = datetime(2019,6,6,0,0)
end_dt =  datetime(2019,6,8,0,0)
list_IDs = get_list_IDs(start_dt, end_dt, config.x_length, config.y_length, filter_no_rain = config.filter_no_rain)
print('Samples in training set:')
print(len(list_IDs))

generator = DataGenerator(list_IDs, batch_size=config.batch_size,
                          x_seq_size=config.x_length, y_seq_size=config.y_length,
                          norm_method=config.norm_method, load_from_npy=False)

# Get images to visualize GAN performance
# val data:
start_dt = datetime(2019,6,6,0,0)
end_dt =  datetime(2019,6,8,0,0)
list_IDs = get_list_IDs(start_dt, end_dt, config.x_length, config.y_length,
                        filter_no_rain=config.filter_no_rain)
generator_val = DataGenerator(list_IDs, batch_size=config.batch_size,
                              x_seq_size=config.x_length, y_seq_size=config.y_length,
                              norm_method=config.norm_method, load_from_npy=False)
x_test = []
y_test = []
for xs, ys in generator_val:
    x_test.append(xs)
    y_test.append(ys)
x_test = np.array(x_test)
y_test = np.array(y_test)
print('Samples in visualize validation set:')
print(len(list_IDs))


# Initialize model
#model = GAN(rnn_type = config.rnn_type, x_length = config.x_length, y_length = config.y_length, architecture = config.g_model)
#model.compile(lr_g = config.lr_g, lr_d = config.lr_d)
model = build_generator(architecture=config.architecture, rnn_type=config.rnn_type, relu_alpha=0.2,
			x_length = config.x_length, y_length = config.y_length)
opt = tf.keras.optimizers.Adam(learning_rate=config.lr_g)
model.compile(loss='mse', metrics=['mse', 'mae'])
history = model.fit(generator, epochs = config.epochs,
                    callbacks = [WandbCallback(),
                                 logger.ImageLogger(x_test,y_test)
                                 ])
