import tensorflow as tf
import numpy as np
import netCDF4
import os
import sys
sys.path.insert(0,'..')
from ConvGRU2D import ConvGRU2D
import config

def get_mask_y():
    '''
    The Overeem images are masked. Only values near the netherlands are kept.
    The model output should also be masked, such that the output of the masked values becomes zero.
    This function returns the approriate mask to mask the output
    '''
    
    path_mask = 'mask.npy'

    if os.path.isfile(path_mask):
        mask = np.load(path_mask)
    else:
        # Get the mask for the input data
        y_path = config.dir_aart
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
def convRNN_block(x, filters, kernel_size, strides, rnn_type='GRU', padding='same', return_sequences=True, 
                  name=None, leakyrelu_alpha=0.2, ):
    if rnn_type == 'GRU':
        x = ConvGRU2D.ConvGRU2D(name=name, filters=filters, kernel_size=kernel_size, 
                                              strides=strides,
                                              padding=padding, 
                                              return_sequences=return_sequences)(x)   
    if rnn_type == 'LSTM':
        x = tf.keras.layers.ConvLSTM2D(name=name, filters=filters, kernel_size=kernel_size, 
                                              strides=strides,
                                              padding=padding, 
                                              return_sequences=return_sequences)(x)   
    x = tf.keras.layers.LeakyReLU(leakyrelu_alpha)(x)
    return x


def encoder(x, rnn_type):
  # Downsample 1a
  x = tf.keras.layers.Conv3D(filters=8, kernel_size=(1,3,3), strides=(1,2,2), padding='same', name='Downsample1a')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # Downsample 1b
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2,2), padding='same', name='Downsample1b')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)   

  # RNN block 1
  x = convRNN_block(x, rnn_type=rnn_type, filters=64, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}1a'.format(rnn_type))
  
  x = convRNN_block(x, rnn_type=rnn_type, filters=64, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}1b'.format(rnn_type))
    
  # Downsample 2
  x = tf.keras.layers.Conv2D( name='Downsample2', filters=64, kernel_size=(5, 5), strides=(3,3), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # RNN block 2
  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}2a'.format(rnn_type))

  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}2b'.format(rnn_type))
    
  # Downsample 3
  x = tf.keras.layers.Conv2D( name='Downsample3', filters=192, kernel_size=(3, 3), strides=(2,2), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # RNN block 2
  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}3a'.format(rnn_type))
  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}3b'.format(rnn_type))
  return x

def decoder(x, rnn_type):
  # Decoder block 1
  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_1a'.format(rnn_type))
  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_1b'.format(rnn_type))

  # Upsample 
  x = tf.keras.layers.Conv3DTranspose( name='Upsample1', filters=192, kernel_size=(1,4, 4), strides=(1,2,2), padding='same')(x)
  x = tf.keras.layers.Cropping3D(cropping=(0, 0, (1,0)))(x)
    
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # Decoder block2
  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_2a'.format(rnn_type))

  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(5, 5), 
                                              strides=(1,1), name='Conv{}_decoder_2b'.format(rnn_type))

  # Upsample 
  x = tf.keras.layers.Conv3DTranspose( name='Upsample2', filters=192, kernel_size=(1,5, 5), strides=(1,3,3), padding='same')(x)
  x = tf.keras.layers.Cropping3D(cropping=(0, 0, 1))(x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  # Decoder block3
  x = convRNN_block(x, rnn_type= rnn_type, filters=64, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_3a'.format(rnn_type))

  x = convRNN_block(x, rnn_type= rnn_type, filters=64, kernel_size=(5, 5), 
                                              strides=(1,1), return_sequences = False, name='Conv{}_decoder_3b'.format(rnn_type))

  # Upsample to target resolution
  x = tf.keras.layers.Conv2DTranspose( name='Upsample3', filters=8, kernel_size=(5, 5), strides=(2,2), padding='same')(x)
  
  x = tf.keras.layers.Conv2DTranspose( name='Conv', filters=1, kernel_size=1, strides=1, padding='same')(x) 
    
  x = tf.keras.layers.Reshape(target_shape=(1,384, 350, 1))(x)
  return x  


def build_generator(rnn_type):
    input_seq = tf.keras.Input(shape=(5, 768, 700, 1))

    x = encoder(input_seq, rnn_type)
    x = decoder(x, rnn_type)
    
    # Apply mask to output
    output = tf.keras.layers.Multiply(name='Mask')([x, get_mask_y()])
    
    model = tf.keras.Model(inputs=input_seq, outputs=output, name='Generator')
    return model

def build_discriminator():
    input_seq = tf.keras.Input(shape=(1, 384, 350, 1))
    
    # Conv1
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(3,3), padding='same', name='Conv1')(input_seq)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    #Conv2
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding='same', name='Conv2')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    
    # Conv3
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding='same', name='Conv3')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
     
    # Conv4
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding='same', name='Conv4')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
        
    # Dense
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
    model = tf.keras.Model(inputs=input_seq, outputs=output, name='Discriminator')
    return model

class GAN(tf.keras.Model):
    def __init__(self, rnn_type='GRU'):
        super(GAN, self).__init__()      
        self.discriminator = build_discriminator()
        self.generator = build_generator(rnn_type)
 
    
    def compile(self, optimizer='adam'):
        super(GAN, self).compile()
        
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)   
        
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.mse_metric = tf.keras.metrics.Mean(name="mse")
    
    def call(self, x):
        """Run the model."""
        y_pred = self.generator(x)
        return y_pred

    
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, batch):
        xs, ys = batch
        batch_size = tf.shape(xs)[0]

        # Decode them to fake images
        generated_images = self.generator(xs)

        # Combine them with real images
        combined_images = tf.concat([generated_images, ys], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Add random noise to the labels - important trick!
        #labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

      
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(xs))
            g_loss_gan = self.loss_fn(misleading_labels, predictions)
            g_loss_mse = self.loss_mse(ys, predictions)
            g_loss = g_loss_gan + g_loss_mse
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss_gan)
        self.mse_metric.update_state(g_loss_mse)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "mse_loss": self.mse_metric.result()
        }
    
