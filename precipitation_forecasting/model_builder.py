import tensorflow as tf
import numpy as np
import netCDF4
import os

def get_mask_y():
    '''
    The Overeem images are masked. Only values near the netherlands are kept.
    The model output should also be masked, such that the output of the masked values becomes zero.
    This function returns the approriate mask to mask the output
    '''
    
    path_mask = '/nobackup/users/schreurs/project_GAN/KNMI_Internship_GANs/precipitation_forecasting/mask.npy'

    if os.path.isfile(path_mask):
        mask = np.load(path_mask)
    else:
        # Get the mask for the input data
        y_path = '/nobackup/users/schreurs/project_GAN/dataset_aart'
        # The mask is the same for all radar scans, so simply chose a random one to get the mask
        path = y_path + '/RAD_NL25_RAC_MFBS_EM_5min_201901010000.nc'

        with netCDF4.Dataset(path, 'r') as f:
            rain = f['image1_image_data'][:].data
            mask = (rain != 65535)
        mask = mask.astype(float)
        mask = np.expand_dims(mask, axis=-1)
        mask = crop_center(mask)
        np.save(path_mask,mask)
    return mask

def crop_center(img,cropx=350,cropy=384):
    # batch size, sequence, height, width, channels
     # Only change height and width
    _, y,x, _ = img.shape
    startx = 20+x//2-(cropx//2)
    starty = 40+y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx:,]


# Based upon the paper by Tian. Used convLSTM instead of ConvGRU for now as the latter is not available in keras. 
# This can later still be implemented.

def encoder(x):
  # Downsample 1a
  x = tf.keras.layers.Conv3D(filters=8, kernel_size=(1,3,3), strides=(1,2,2), padding='same', name='Downsample1a')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # Downsample 1b
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2,2), padding='same', name='Downsample1b')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)   

  # RNN block 1
  x = tf.keras.layers.ConvLSTM2D(name='ConvLSTM1a', filters=64, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='ConvLSTM1b', filters=64, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # Downsample 2
  x = tf.keras.layers.Conv2D( name='Downsample2', filters=64, kernel_size=(5, 5), strides=(3,3), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # RNN block 2
  x = tf.keras.layers.ConvLSTM2D(name='ConvLSTM2a', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='ConvLSTM2b', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)  

  # Downsample 3
  x = tf.keras.layers.Conv2D( name='Downsample3', filters=192, kernel_size=(3, 3), strides=(2,2), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # RNN block 2
  x = tf.keras.layers.ConvLSTM2D(name='ConvLSTM3a', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='ConvLSTM3b', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)  
  return x

def decoder(x):
  # Decoder block 1
  x = tf.keras.layers.ConvLSTM2D(name='Decoder_Block1_ConvLSTM_1', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='Decoder_Block1_ConvLSTM_2', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)  

  # Upsample 
  x = tf.keras.layers.Conv3DTranspose( name='Upsample1', filters=192, kernel_size=(1,4, 4), strides=(1,2,2), padding='same')(x)
  x = tf.keras.layers.Cropping3D(cropping=(0, 0, (1,0)))(x)
    
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # Decoder block2
  x = tf.keras.layers.ConvLSTM2D(name='Decoder_Block2_ConvLSTM_1', filters=192, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='Decoder_Block2_ConvLSTM_2', filters=192, kernel_size=(5, 5), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)  

  # Upsample 
  x = tf.keras.layers.Conv3DTranspose( name='Upsample2', filters=192, kernel_size=(1,5, 5), strides=(1,3,3), padding='same')(x)
  x = tf.keras.layers.Cropping3D(cropping=(0, 0, 1))(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='Decoder_Block3_ConvLSTM_1', filters=64, kernel_size=(3, 3), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=True)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  x = tf.keras.layers.ConvLSTM2D(name='Decoder_Block3_ConvLSTM_2', filters=64, kernel_size=(5, 5), 
                                          strides=(1,1),
                                          padding='same', 
                                          return_sequences=False)(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)  

  # Upsample to target resolution
  x = tf.keras.layers.Conv2DTranspose( name='Upsample3', filters=8, kernel_size=(5, 5), strides=(2,2), padding='same')(x)
  
  x = tf.keras.layers.Conv2DTranspose( name='Conv', filters=1, kernel_size=1, strides=1, padding='same')(x) 
    
  x = tf.keras.layers.Reshape(target_shape=(1,384, 350, 1))(x)
  return x  


def build_generator():
    input_seq = tf.keras.Input(shape=(5, 768, 700, 1))

    x = encoder(input_seq)
    x = decoder(x)
    
    # Apply mask to output
    output = tf.keras.layers.Multiply(name='Mask')([x, get_mask_y()])
    
    model = tf.keras.Model(inputs=input_seq, outputs=output, name='Generator')
    return model




