from batchcreator import DataGenerator, get_list_IDs
from datetime import datetime
from model_builder import GAN, build_generator
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

print('Starting test run')
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


run = wandb.init(project='precipitation-forecasting',
		config={
			'batch_size' : 1,
			'epochs': 5,
			'x_length': 6,
			'y_length': 1,
			'rnn_type': 'GRU',
			'filter_no_rain': True,
			'dataset': '2019-06-4 till 2019-06-12',
			'g_model': 'AENN',
			'model': 'GAN'
		})

config = wandb.config
batch_size = config.batch_size
# Create list of IDs to retrieve
x_seq_size= config.x_length
y_seq_size= config.y_length

start_dt = datetime(2019,6,4,0,0)
end_dt =  datetime(2019,6,12,0,00)
# Exclude samples were there is no rain in the input
filter_no_rain = config.filter_no_rain
list_IDs = get_list_IDs(start_dt, end_dt, x_seq_size, y_seq_size, filter_no_rain = filter_no_rain)
print('Samples in training set:')
print(len(list_IDs))


# Initialize model
#model = build_generator(rnn_type='GRU', relu_alpha=0.2, x_length=6,  y_length=1, architecture='AENN')
model = GAN(rnn_type=config.rnn_type, x_length=x_seq_size, y_length=y_seq_size, architecture = config.g_model)
model.compile()

# Create generator and train the model:
generator = DataGenerator(list_IDs, batch_size=batch_size, x_seq_size=x_seq_size, y_seq_size=y_seq_size, norm_method='minmax', load_from_npy=False)
history = model.fit(generator, epochs=config.epochs, callbacks = [WandbCallback()])
