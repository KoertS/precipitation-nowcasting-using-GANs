import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, 
                 img_dim = (765, 700, 1), x_seq_size=3, y_seq_size=1, shuffle=True,
                x_path = '/nobackup/users/schreurs/project_GAN/dataset_radar_np',
                y_path = '/nobackup/users/schreurs/project_GAN/dataset_aart_np',
                normalize_y=False, crop_y=True, pad_x=True):
        'Initialization'
        self.inp_shape = (x_seq_size, *img_dim)
        self.out_shape = (y_seq_size, *img_dim)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.x_path = x_path
        self.y_path = y_path
        self.normalize_y = normalize_y
        

        self.crop_y = crop_y
        self.pad_x = pad_x
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.inp_shape))
        y = np.empty((self.batch_size, *self.out_shape))
        
        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            x_IDs, y_IDs = IDs
            
            # Store input image(s)
            for c in range(self.inp_shape[0]):
                X[i,c] = self.load_x(x_IDs[c])

            # Store target image(s)
            for c in range(self.out_shape[0]):
                y[i,c] = self.load_y(y_IDs[c])
                
        if self.pad_x:
            X = self.pad_along_axis(X)
        if self.normalize_y:
            y = self.zscore(y)
        
        if self.crop_y:
            y = self.crop_center(y)
        return X, y
    
    def load_x(self, x_ID):
        path = self.x_path + '/{}.npy'.format(x_ID)
      
        rain = np.load(path)
        # set masked values to 0
        rain[rain == 65535] = 0
        # Expand dimensions from (w,h) to (w,h,c=1)
        rain = np.expand_dims(rain, axis=-1)
        
        return rain
        
    def load_y(self, y_ID):
        path = self.y_path + '/{}.npy'.format(y_ID)

        rain = np.load(path)
        # set masked values to 0
        # Note that the data was converted to integers by multiplying with 100
        rain[rain == 65535*100] = 0
        # Expand dimensions from (w,h) to (w,h,c=1)
        rain = np.expand_dims(rain, axis=-1)
            
        return rain
        
    def zscore(self, x):
        MEAN = 0.7740296547051635
        STD = 37.88184326601481

        return (x-MEAN)/STD
    
    def crop_center(self, img,cropx=350,cropy=384):
        # batch size, sequence, height, width, channels
        # Only change height and width
        _,_, y,x, _ = img.shape
        startx = 20+x//2-(cropx//2)
        starty = 40+y//2-(cropy//2)    
        return img[:,:,starty:starty+cropy,startx:startx+cropx:,]
    
    def pad_along_axis(self, array, pad_size = 3, axis = 2):
        '''
        Pad input to be divisible by 4. 
        height of 765 to 768
        '''
        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)



def get_list_IDs(start_dt, end_dt,x_seq_size=5,y_seq_size=1, filter_no_rain=False):
    '''
    This function returns filenames between the a starting date and end date. 
    The filenames are packed into input and output arrays.
    start_dt: starting date time 
    end_dt: end date time
    x_seq_size: size of the input sequence
    y_seq_size: size of the output sequence
    filter_no_rain: boolean that indicates wether to discard input data were a scan has no rain
    '''
    label_dir = '/nobackup/users/schreurs/project_GAN/rtcor_rain_labels'
    
    # Create list of IDs to retrieve
    dts = np.arange( start_dt, end_dt, timedelta(minutes=5*x_seq_size)).astype(datetime)
    # Convert to filenames
    list_IDs = []
    for dt in dts:
        list_ID = xs, ys =  get_filenames_xy(dt,x_seq_size,y_seq_size)
        
        if filter_no_rain:  
            try:
                has_rain = not any([np.load(label_dir+ '/{}.npy'.format(file)) for file in xs])
            except:
                has_rain = False
    
            if has_rain:
                list_IDs.append(list_ID)
        else:
            list_IDs.append(list_ID)
    return list_IDs 

def get_filenames_xy(dt, x_size=5, y_size=1):
    '''
    Returns the filenames of the input x and target y. 
    dt: datetime of sample (year month day hour minute)
    x_size: how many samples to take before datetime
    y_size: how many samples to take after datetime 
    '''
    xs = []
    for i in range(x_size,0,-1):
        dt_i = dt - i*timedelta(minutes=5)
        ts = '{:%Y%m%d%H%M}'.format(dt_i)
        xs.append(ts)
        
    ys = []
    for i in range(0,y_size,1):
        dt_i = dt + i*timedelta(minutes=5)
        ts = '{:%Y%m%d%H%M}'.format(dt_i)
        ys.append(ts)
    return xs,ys