import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
from netCDF4 import Dataset
import config


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, x_seq_size=6, 
                 y_seq_size=24, shuffle=True, load_from_npy=True,
                norm_method=None, crop_y=True, pad_x=True,
                downscale256 = False, convert_to_dbz = False):
        '''
        list_IDs: pair of input and target filenames
        batch_size: size of batch to generate
        x_seq_size: length of input sequence
        y_seq_size: length of target sequence 
        shuffle: if true than shuffle the batch
        x_path: path to input data
        y_path: path to target data
        load_from_npy: specifies if the generator should load the nc/h5 files and convert to numpy or load the numpy files directly
        norm_method: string that states what normalization method to use: 'minmax' or 'znorm'
        crop_y: if true then crop around the netherlands, halving the image size
        pad_x: adds 3 empty rows to input data to make it divisible by 2
        downscale256: If true uses bilinear interpolation to downscale input and output to 256x256
        convert_to_dbz: If true the rain values (mm/h) will be transformed into dbz
        '''
        img_dim = (765, 700, 1)
        self.inp_shape = (x_seq_size, *img_dim)
        self.out_shape = (y_seq_size, *img_dim)
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_from_npy = load_from_npy

        
        self.x_path = config.dir_rtcor
        self.y_path = config.dir_aart 
        if self.load_from_npy:
            self.x_path = config.dir_rtcor_npy
            self.y_path = config.dir_aart_npy
             
        # Normalize
        self.norm_method = norm_method
        if norm_method and norm_method != 'minmax' and norm_method != 'zscore' and norm_method != 'minmax_tanh':
            print('Unknown normalization method. Options are \'minmax\' , \'zscore\' and \'minmax_tanh\'')
            print('Normalization method has been set to None')
            self.norm_method = None
            
        self.crop_y = crop_y
        self.pad_x = pad_x
        self.downscale256 = downscale256
        if downscale256:
            self.crop_y = self.pad_x = False
        self.convert_to_dbz = convert_to_dbz

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
                
        if self.norm_method == 'zscore':
            y = self.zscore(y)
        if self.norm_method == 'minmax':
            X = minmax(X, convert_to_dbz=self.convert_to_dbz)
            y = minmax(y, convert_to_dbz=self.convert_to_dbz)
        if self.norm_method == 'minmax_tanh':
            X = minmax(X, tanh=True, convert_to_dbz=self.convert_to_dbz)
            y = minmax(y, tanh=True, convert_to_dbz=self.convert_to_dbz)
        if self.pad_x:
            X = self.pad_along_axis(X, axis=2, pad_size=3)
        if self.crop_y:
            y = self.crop_center(y)
        if self.downscale256:
            # First make the images square size
            X = self.pad_along_axis(X, axis=2, pad_size=3)
            X = self.pad_along_axis(X, axis=3, pad_size=68)
            y = self.crop_center(y, cropx=384, cropy=384)
            
            X =  np.array([tf.image.resize(x, (256, 256)) for x in X])
            y =  np.array([tf.image.resize(y_i, (256, 256)) for y_i in y])
        return X, y
    
    def load_h5(self, path):
        '''
        The orginial input images are stored in .h5 files. 
        This function loads them and converts them to numpy arrays
        '''
        radar_img = None
        with h5py.File(path, 'r') as f:
            try:
                radar_img = f['image1']['image_data'][:]

                ## Set pixels out of image to 0
                out_of_image = f['image1']['calibration'].attrs['calibration_out_of_image']
                radar_img[radar_img == out_of_image] = 0
                # Sometimes 255 or other number (244) is used for the calibration
                # for out of image values, so also check the first pixel
                radar_img[radar_img == radar_img[0][0]] = 0
            except:
                print("Error: could not read image1 data, file {}".format(path))
        return radar_img
    
    def load_nc(self, path, as_int=True):
        '''
        The target images are stored as .nc files. 
        This function loads and converts the images to numpy arrays
        '''
        radar_img = None
        with Dataset(path, mode='r') as f:
            try:
                radar_img = f['image1_image_data'][:][0].data
                mask = f['image1_image_data'][:][0].mask
                
                # apply mask:
                radar_img = radar_img * ~mask
                
                if as_int:
                    radar_img *=100
                    radar_img = radar_img.astype(int)
            except:
                print("Error: could not read image data, file {}".format(path))
        return radar_img
   
        
    def load_x(self, x_ID):
        if self.load_from_npy:
            path = self.x_path + '{}.npy'.format(x_ID)
            x = np.load(path)
        else:
            dt = datetime.strptime(x_ID, '%Y%m%d%H%M')
            path = self.x_path +  '{Y}/{m:02d}/{prefix}{ts}.h5'.format(Y=dt.year, m=dt.month, prefix = config.prefix_rtcor,  ts=x_ID)
            x = self.load_h5(path)   
            
        # set masked values to 0
        x[x == 65535] = 0
        # Expand dimensions from (w,h) to (w,h,c=1)
        x = np.expand_dims(x, axis=-1)
        
        return x
        
    def load_y(self, y_ID):
        if self.load_from_npy:
            path = self.y_path + '{}.npy'.format(y_ID)
            y = np.load(path)
        else:
            year = y_ID[:4]
            path = self.y_path + year + '/' + config.prefix_aart + y_ID +'.nc'
            y = self.load_nc(path)
        # set masked values to 0
        # Note that the data was converted to integers by multiplying with 100
        y[y == 65535*100] = 0
        # Expand dimensions from (w,h) to (w,h,c=1)
        y = np.expand_dims(y, axis=-1)
        
        return y
        
    def zscore(self, x):
        MEAN = 0.7740296547051635
        STD = 37.88184326601481

        return (x-MEAN)/STD
    
    
    def crop_center(self, img,cropx=350,cropy=384):
        # batch size, sequence, height, width, channels
        # Only change height and width
        _,_,y,x, _ = img.shape
        startx = 20+x//2-(cropx//2)
        starty = 40+y//2-(cropy//2)    
        return img[:,:,starty:starty+cropy,startx:startx+cropx:,]
    
    def pad_along_axis(self, array, pad_size = 3, axis = 2):
        '''
        Pad input to be divisible by 2. 
        height of 765 to 768
        '''
        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return tf.pad(array, pad_width=npad, mode='constant', constant_values=0)

def minmax(x, tanh=False, undo=False, convert_to_dbz = False):
    '''
    Performs minmax scaling to scale the images to range of 0 to 1.
    If tanh is True than scale to -1 to 1 as tanh is used for activation function generator
    '''
    # pixel values are in 0.01mm
    # define max intensity as 100mm
    MIN = 0
    MAX = 10000
    
    if convert_to_dbz:
        MAX = 55
        x = r_to_dbz(x)
    if not undo:
        # Set values over 100mm/h to 100mm/h
        x = tf.clip_by_value(x, MIN, MAX)
        if tanh:
            x = (x - MIN - MAX/2)/(MAX/2 - MIN) 
        else:
            x = (x - MIN)/(MAX- MIN)
    else:
        if tanh:
            x = x*(MAX/2 - MIN) + MIN + MAX/2
        else:
            x = x*(MAX - MIN) + MIN
    return x

def r_to_dbz(r):
    '''
    Convert mm/h to dbz
    '''
    # Pixel values are in 0.01mm
    # Convert r to be in mm/h not 0.01mm/h
    r = r / 100
    return 10 * np.log10(200*r**(8/5)+1) 

def get_list_IDs(start_dt, end_dt,x_seq_size=6,y_seq_size=1, filter_no_rain=None):
    '''
    This function returns filenames between the a starting date and end date. 
    The filenames are packed into input and output arrays.
    start_dt: starting date time 
    end_dt: end date time
    x_seq_size: size of the input sequence
    y_seq_size: size of the output sequence
    filter_no_rain: If None than use all samples, 
        if sum30mm than filter less than 30mm per image, 
        if avg0.01mm than filter out samples with average rain below 0.01mm/pixel
    '''
    
    
    if filter_no_rain == 'sum30mm':
        label_dir = config.dir_labels
    elif filter_no_rain == 'avg0.01mm':
        label_dir = config.dir_labels_heavy 
    elif filter_no_rain:
        print('Error: unkown filter_no_rain argument {}. Options are \'sum30mm\' and \'avg0.01mm\'. ')
        print('Setting filtering to \'sum30mm\''.format(filter_no_rain))
        label_dir = label_dir = config.dir_labels
        
    # Create list of IDs to retrieve
    dts = np.arange( start_dt, end_dt, timedelta(minutes=5*x_seq_size)).astype(datetime)
    # Convert to filenames
    list_IDs = []
    for dt in dts:
        list_ID = xs, ys =  get_filenames_xy(dt,x_seq_size,y_seq_size)

        if filter_no_rain:
            try:
                has_rain = all([np.load(label_dir+ '{}/{}/{}.npy'.format(file[:4], file[4:6], file)) for file in xs])
            except Exception as e:
                print(e)
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