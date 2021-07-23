from datetime import datetime
import tensorflow as tf
import numpy as np

# Add parent directory to system path in order to import custom modules
import sys
sys.path.append('../')
sys.path.append('../../')

from model_builder import GAN
from batchcreator import DataGenerator, get_list_IDs

# Set hyperparameters
x_length = 6
y_length = 3
filter_no_rain = 'avg0.01mm'
architecture = 'AENN'
l_adv = 0.003
l_rec = 1
g_cycles = 3
label_smoothing = 0.2
lr_g = 0.003
lr_d = 0.001

# Loads preproccesed files:
load_prep = True
# Set rtcor as target (instead of aart)
y_is_rtcor= True

# Either select files by defining a time period or load a premade list of filenames:
#start_dt = datetime(2019,6,1,0,0)
#end_dt =  datetime(2019,7,1,0,0)
#list_IDs = get_list_IDs(start_dt, end_dt, 6, 3, filter_no_rain=filter_no_rain, y_interval=30)

# Select filename between a start date and end data
# get_list_IDs function can take long time on RU server (limited cpu power I think?)
# other option is to load predefined list of filenames, for example:
list_IDs = np.load('../datasets/train2015_2018_3y_30m.npy', allow_pickle=  True)
list_IDs = list_IDs[:100] # reduce dataset size for testing purposes

model = GAN(rnn_type='GRU', x_length=x_length,
            y_length=y_length, architecture=architecture, relu_alpha=.2,
           l_adv = l_adv, l_rec = l_rec, g_cycles=3, label_smoothing=label_smoothing,
           norm_method = 'minmax', downscale256 = True, rec_with_mae= False,
           batch_norm = False)
model.compile(lr_g = lr_g, lr_d = lr_d)

generator = DataGenerator(list_IDs, batch_size=8, x_seq_size=x_length, 
                                       y_seq_size=y_length, load_prep=load_prep, y_is_rtcor= y_is_rtcor)
hist = model.fit(generator, epochs=1)
