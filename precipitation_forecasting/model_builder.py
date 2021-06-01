import tensorflow as tf
import numpy as np
import netCDF4
import os
import config
import sys
sys.path.insert(0,config.path_project)
sys.path.insert(0,'..')
from ConvGRU2D import ConvGRU2D
from tensorflow.keras.optimizers import Adam

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
        path = y_path + config.prefix_aart + '201901010000.nc'

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
                  name=None, relu_alpha=0.2):
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
    x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
    return x

def conv_block(x, filters, kernel_size, strides, padding='same', name=None, relu_alpha=0.2, transposed = False, output_layer=False):
    layer =  tf.keras.layers.Conv2D
    if transposed:
        layer = tf.keras.layers.Conv2DTranspose
 
    x = layer(name=name, filters=filters, kernel_size=kernel_size, 
                                              strides=strides, padding=padding)(x)
    
    if output_layer:
        x = tf.keras.activations.linear(x)
    else:
        x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
    return x

def encoder(x, rnn_type, relu_alpha):
  # Downsample 1a
  x = tf.keras.layers.Conv3D(filters=8, kernel_size=(1,3,3), strides=(1,2,2), padding='same', name='Downsample1a')(x)
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)

  # Downsample 1b
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2,2), padding='same', name='Downsample1b')(x)
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)   

  # RNN block 1
  x = convRNN_block(x, rnn_type=rnn_type, filters=64, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}1a'.format(rnn_type), relu_alpha=relu_alpha)
  
  x = convRNN_block(x, rnn_type=rnn_type, filters=64, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}1b'.format(rnn_type), relu_alpha=relu_alpha)
    
  # Downsample 2
  x = tf.keras.layers.Conv2D( name='Downsample2', filters=64, kernel_size=(5, 5), strides=(3,3), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)

  # RNN block 2
  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}2a'.format(rnn_type),relu_alpha=relu_alpha)

  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}2b'.format(rnn_type),relu_alpha=relu_alpha)
    
  # Downsample 3
  x = tf.keras.layers.Conv2D( name='Downsample3', filters=192, kernel_size=(3, 3), strides=(2,2), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)

  # RNN block 2
  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}3a'.format(rnn_type),relu_alpha=relu_alpha)
  x = convRNN_block(x, rnn_type=rnn_type, filters=192, kernel_size=(3, 3), 
                                          strides=(1,1), name='Conv{}3b'.format(rnn_type),relu_alpha=relu_alpha)
  return x

def decoder(x, rnn_type, relu_alpha):
  # Decoder block 1
  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_1a'.format(rnn_type),relu_alpha=relu_alpha)
  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_1b'.format(rnn_type),relu_alpha=relu_alpha)

  # Upsample 
  x = tf.keras.layers.Conv3DTranspose( name='Upsample1', filters=192, kernel_size=(1,4, 4), strides=(1,2,2), padding='same')(x)
  x = tf.keras.layers.Cropping3D(cropping=(0, 0, (1,0)))(x)
    
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)

  # Decoder block2
  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_2a'.format(rnn_type),relu_alpha=relu_alpha)

  x = convRNN_block(x, rnn_type= rnn_type, filters=192, kernel_size=(5, 5), 
                                              strides=(1,1), name='Conv{}_decoder_2b'.format(rnn_type),relu_alpha=relu_alpha)

  # Upsample 
  x = tf.keras.layers.Conv3DTranspose( name='Upsample2', filters=192, kernel_size=(1,5, 5), strides=(1,3,3), padding='same')(x)
  x = tf.keras.layers.Cropping3D(cropping=(0, 0, 1))(x)
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)

  # Decoder block3
  x = convRNN_block(x, rnn_type= rnn_type, filters=64, kernel_size=(3, 3), 
                                              strides=(1,1), name='Conv{}_decoder_3a'.format(rnn_type),relu_alpha=relu_alpha)

  x = convRNN_block(x, rnn_type= rnn_type, filters=64, kernel_size=(5, 5), 
                                              strides=(1,1), return_sequences = False, 
                    name='Conv{}_decoder_3b'.format(rnn_type), relu_alpha=relu_alpha)

  # Upsample to target resolution
  x = tf.keras.layers.Conv2DTranspose( name='Upsample3', filters=8, kernel_size=(5, 5), strides=(2,2), padding='same')(x)
  x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
  x = tf.keras.layers.Conv2DTranspose( name='Conv', filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(x) 
    
  x = tf.keras.layers.Reshape(target_shape=(1,384, 350, 1))(x)
  return x  

def discriminator_Tian(x, relu_alpha):
    # Conv1
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(3,3), padding='same', name='Conv1')(input_seq)
    x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
    
    #Conv2
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding='same', name='Conv2')(x)
    x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
    
    
    # Conv3
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding='same', name='Conv3')(x)
    x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
     
    # Conv4
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding='same', name='Conv4')(x)
    x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
        
    # Dense
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x) 
    return output

def generator_AENN(x, rnn_type='GRU', relu_alpha=0.2, x_length=6, y_length=1):
    ''' 
    This generator uses similar architecture as in AENN.
    An extra encoder layer was added to downsample the input image.
    Furthermore the second layer uses stride of 3 
    instead of 2 to get to desired shape of 128x128
    AENN paper: jing2019
    '''
    # Add padding to make square image
    x = tf.keras.layers.ZeroPadding3D(padding=(0,0,34))(x)
        
    # Encoder:    
    x = conv_block(x, filters = 32, kernel_size=5, strides = 2, 
                      relu_alpha = relu_alpha)
    x = conv_block(x, filters = 32, kernel_size=5, strides = 3, 
                      relu_alpha = relu_alpha)
    x = conv_block(x, filters = 64, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha) 
    x = conv_block(x, filters = 128, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha) 

    # RNN part:
    x = convRNN_block(x, filters = 128, kernel_size=3, strides = 1, 
                      relu_alpha = relu_alpha,  rnn_type=rnn_type) 
    x = convRNN_block(x, filters = 128, kernel_size=3, strides = 1, 
                      relu_alpha = relu_alpha,  rnn_type=rnn_type, return_sequences=False) 
    
    # Decoder:
    x = conv_block(x, filters = 64, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, transposed = True)
    
    x = conv_block(x, filters = 32, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, transposed = True)
    x = conv_block(x, filters = y_length, kernel_size=3, strides = 3, 
                      output_layer=True, transposed = True)
    
    # Convert to predictions
    # Crop to fit output shape
    x = tf.keras.layers.Cropping2D((0,17))(x)
    
    # Apply mask to output
    x = tf.keras.layers.Reshape(target_shape=(y_length,384, 350, 1))(x)
    
    output = x 
    return output

def discriminator_AENN(x, relu_alpha):
    # Add padding to make square image
    x = tf.keras.layers.ZeroPadding3D(padding=(0,0,17))(x)  
    
    x = conv_block(x, filters = 32, kernel_size=5, strides = 3, 
                      relu_alpha = relu_alpha)   
    x = conv_block(x, filters = 64, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha)        
    x = conv_block(x, filters = 128, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha)
    x = conv_block(x, filters = 256, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha) 
    x = conv_block(x, filters = 512, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha)
    x = tf.keras.layers.AveragePooling3D(pool_size=(1,8,8))(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x) 
    
    return output

def build_generator(rnn_type, relu_alpha, x_length=6, y_length=1, architecture='Tian'):
    inp = tf.keras.Input(shape=(x_length, 768, 700, 1))

    if architecture == 'Tian':
        x = encoder(inp, rnn_type, relu_alpha)
        output = decoder(x, rnn_type, relu_alpha)        
    elif architecture == 'AENN':
        output = generator_AENN(inp, rnn_type, relu_alpha, x_length, y_length)
    else:
        raise Exception('Unkown architecture {}. Option are: Tian, AENN'.format(architecture))
    # Mask pixels outside Netherlands  
    mask = tf.constant(get_mask_y(), 'float32')
    masked_output = tf.keras.layers.Lambda(lambda x: x * mask, name='Mask')(output)
   # If tanh activation:
   # output = tf.keras.layers.subtract([output, 1-mask])  
    
    model = tf.keras.Model(inputs=inp, outputs=masked_output, name='Generator')
    return model

def build_discriminator(relu_alpha, y_length, architecture = 'Tian'):
    inp = tf.keras.Input(shape=(y_length, 384, 350, 1))
    
    if architecture == 'Tian':
        output = discriminator_Tian(inp, relu_alpha)
    elif architecture == 'AENN':
        output = discriminator_AENN(inp, relu_alpha)
    else:
        raise Exception('Unkown architecture {}. Option are: Tian, AENN'.format(architecture))
        
    model = tf.keras.Model(inputs=inp, outputs=output, name='Discriminator')
    return model

class GAN(tf.keras.Model):
    def __init__(self, rnn_type='GRU', x_length=6, y_length=1, relu_alpha=0.2, architecture='Tian', l_g = 1, l_mse = 0.01,
                g_cycles=1, noise_labels = 0):
        '''
        rnn_type: type of recurrent neural network can be LSTM or GRU
        x_length: length of input sequence
        y_length: length of output sequence
        relu_alpha: slope of leaky relu layers
        architecture: either 'Tian' or 'AENN'
        l_g: weight of loss GAN for generator
        l_mse: weight of mse for the generator
        g_cycles: how many cycles to train the generator per train cycle
        noise_labels: if higher than 0, noise is added to the labels
        '''
        super(GAN, self).__init__()      

        self.generator = build_generator(rnn_type, x_length=x_length, 
                                         y_length = y_length, relu_alpha=relu_alpha, 
                                         architecture=architecture)
        self.discriminator = build_discriminator(y_length=y_length, 
                                                 relu_alpha=relu_alpha,
                                                architecture=architecture)
        
        self.l_g = l_g
        self.l_mse = l_mse
        self.g_cycles=g_cycles
        self.noise_labels=noise_labels
 
    def compile(self, lr_g=0.0001, lr_d = 0.0001):
        super(GAN, self).compile()
        
        self.g_optimizer = Adam(learning_rate=lr_g) 
        self.d_optimizer = Adam(learning_rate=lr_d)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.mse_metric = tf.keras.metrics.Mean(name="mse")
        self.d_acc = tf.keras.metrics.BinaryAccuracy(name='d_acc')

    
    def call(self, x):
        """Run the model."""
        y_pred = self.generator(x)
        return y_pred

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.mse_metric, self.d_acc]

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
        labels += self.noise_labels * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        # Update D accuracy metric
        self.d_acc.update_state(labels, predictions)
      
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        for _ in range(self.g_cycles):
            with tf.GradientTape() as tape:
                generated_images = self.generator(xs)
                predictions = self.discriminator(generated_images)
                g_loss_gan = self.loss_fn(misleading_labels, predictions)
                g_loss_mse = self.loss_mse(ys, generated_images)
                g_loss = self.l_g * g_loss_gan  + self.l_mse * g_loss_mse       
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss_gan)
        self.mse_metric.update_state(g_loss_mse)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "mse": self.mse_metric.result(),
            "d_acc": self.d_acc.result()
        } 
    
    def test_step(self, batch):
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
        labels += self.noise_labels * tf.random.uniform(tf.shape(labels))

        predictions = self.discriminator(combined_images)
        d_loss = self.loss_fn(labels, predictions)
        
        # Update D accuracy metric
        self.d_acc.update_state(labels, predictions)
        
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        generated_images = self.generator(xs)
        predictions = self.discriminator(generated_images)
        g_loss_gan = self.loss_fn(misleading_labels, predictions)
        g_loss_mse = self.loss_mse(ys, generated_images)
        g_loss = self.l_g * g_loss_gan  + self.l_mse * g_loss_mse       
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss_gan)
        self.mse_metric.update_state(g_loss_mse)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "mse": self.mse_metric.result(),
            "d_acc": self.d_acc.result()
        } 