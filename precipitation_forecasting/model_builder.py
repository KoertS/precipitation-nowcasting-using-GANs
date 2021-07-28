import tensorflow as tf
import numpy as np
import netCDF4
import os
import config as conf
import sys
sys.path.insert(0,conf.path_project)
sys.path.insert(0,'..')
from ConvGRU2D import ConvGRU2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint
from RepeatVector4D import RepeatVector4D

import batchcreator
from batchcreator import minmax, dbz_to_r

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
    
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}
    
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
        y_path = conf.dir_aartS
        # The mask is the same for all radar scans, so simply chose a random one to get the mask
        path = y_path + '2019/' + conf.prefix_aart + '201901010000.nc'

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
                  name=None, relu_alpha=0.2, wgan = False, batch_norm = False, return_state = False, initial_state=None):
    const = None
    if wgan:
        const = ClipConstraint(0.01)
    
    if rnn_type == 'GRU':
        layer = ConvGRU2D.ConvGRU2D
    if rnn_type == 'LSTM':
        layer = tf.keras.layers.ConvLSTM2D
        
    x= layer(name=name, filters=filters, kernel_size=kernel_size, strides=strides, 
              padding=padding, return_sequences=return_sequences, kernel_constraint=const, 
              return_state = return_state)(x, initial_state = initial_state)
    
    if return_state:
        x, *state  = x
    else:
        state = None
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
    return x, state

def conv_block(x, filters, kernel_size, strides, padding='same', name=None, relu_alpha=0.2, 
               transposed = False, output_layer=False, wgan = False,  batch_norm = False, drop_out = False):
    layer =  tf.keras.layers.Conv2D
    if transposed:
        layer = tf.keras.layers.Conv2DTranspose
        
    const = None
    if wgan:
        const = ClipConstraint(0.01)
        
    conv_layer = layer(name=name, filters=filters, kernel_size=kernel_size, 
                                              strides=strides, padding=padding, kernel_constraint=const)
    x = tf.keras.layers.TimeDistributed(conv_layer)(x)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)    
    if output_layer:
        x = tf.keras.activations.relu(x, max_value=1)
    else:
        x = tf.keras.layers.LeakyReLU(relu_alpha)(x)
        if drop_out:
            x = tf.keras.layers.Dropout(0.2)(x)
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
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(3,3), padding='same', name='Conv1')(x)
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

def generator_AENN(x, rnn_type='GRU', relu_alpha=0.2, x_length=6, y_length=1, norm_method = None, downscale256 = False, 
                  batch_norm = False, num_filters = 32):
    ''' 
    This generator uses similar architecture as in AENN.
    An extra encoder layer was added to downsample the input image.
    Furthermore the second layer uses stride of 3 
    instead of 2 to get to desired shape of 128x128
    AENN paper: jing2019
    '''
    if not downscale256:
        # Add padding to make square image
        x = tf.keras.layers.ZeroPadding3D(padding=(0,0,34))(x)
        
    # Encoder:    
    x = conv_block(x, filters = num_filters, kernel_size=5, strides = 2, 
                      relu_alpha = relu_alpha, batch_norm = batch_norm)
    # If input is not downscaled an extra convolution is needed 
    # to get to same dimensions as AENN network
    if not downscale256:
        x = conv_block(x, filters = num_filters, kernel_size=5, strides = 3, 
                          relu_alpha = relu_alpha, batch_norm = batch_norm)
    x = conv_block(x, filters = num_filters * 2, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, batch_norm = batch_norm) 
    x = conv_block(x, filters = num_filters * 4, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, batch_norm = batch_norm) 

    # RNN part:
    if y_length > 1:
        x, state = convRNN_block(x, filters = num_filters * 4, kernel_size=3, strides = 1, 
                          relu_alpha = relu_alpha,  rnn_type=rnn_type, return_sequences = False,
                          return_state = True, batch_norm = batch_norm) 
        x = RepeatVector4D(y_length)(x)
        x, _ = convRNN_block(x, filters = num_filters * 4, kernel_size=3, strides = 1, 
                          relu_alpha = relu_alpha,  rnn_type=rnn_type, return_sequences= True,
                         initial_state = state, batch_norm = batch_norm) 
    else:
        x, _ = convRNN_block(x, filters = num_filters * 4, kernel_size=3, strides = 1, 
                      relu_alpha = relu_alpha,  rnn_type=rnn_type, batch_norm = batch_norm) 
        x, _ = convRNN_block(x, filters = num_filters * 4, kernel_size=3, strides = 1, 
                      relu_alpha = relu_alpha,  rnn_type=rnn_type, return_sequences = False, batch_norm = batch_norm) 
       
        x = tf.keras.layers.Reshape(target_shape=(y_length,32,32,num_filters * 4))(x)
    # Decoder:
    x = conv_block(x, filters = num_filters * 2, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, transposed = True, batch_norm = batch_norm)
    
    x = conv_block(x, filters = num_filters, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, transposed = True, batch_norm = batch_norm)
    strides_last = 3
    if downscale256:
        strides_last = 2
    x = conv_block(x, filters = 1, kernel_size=3, strides = strides_last, 
                      output_layer=True, transposed = True, batch_norm = batch_norm)
    
    if norm_method and norm_method == 'minmax_tanh':
        x = tf.keras.activations.tanh(x)

    # Crop to fit output shape
    if not downscale256:
        x = tf.keras.layers.Cropping2D((0,17))(x)
    
    output = x 
    return output

def discriminator_AENN(x, relu_alpha,  wgan = False, downscale256 = False, batch_norm = False, drop_out = False):     
    if downscale256:
        strides_first = 2
    else:
        # Add padding to make square image
        x = tf.keras.layers.ZeroPadding3D(padding=(0,0,17))(x)  
        strides_first = 3
        
    x = conv_block(x, filters = 32, kernel_size=5, strides = strides_first, 
                      relu_alpha = relu_alpha, wgan = wgan, batch_norm = batch_norm, drop_out = drop_out)   
    x = conv_block(x, filters = 64, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, wgan = wgan, batch_norm = batch_norm, drop_out = drop_out)        
    x = conv_block(x, filters = 128, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, wgan = wgan, batch_norm = batch_norm, drop_out = drop_out)
    x = conv_block(x, filters = 256, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, wgan = wgan, batch_norm = batch_norm, drop_out = drop_out) 
    x = conv_block(x, filters = 512, kernel_size=3, strides = 2, 
                      relu_alpha = relu_alpha, wgan = wgan, batch_norm = batch_norm, drop_out = drop_out)
    x = tf.keras.layers.AveragePooling3D(pool_size=(1,8,8))(x)
    x = tf.keras.layers.Flatten()(x)
    
    if wgan:
        output = tf.keras.layers.Dense(1, activation='linear')(x) 
    else:
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x) 
    return output

def build_generator(rnn_type, relu_alpha, x_length=6, y_length=1, architecture='Tian', 
                    norm_method = None, downscale256 = False, batch_norm = False, num_filters = 32):
    inp_dim = (768, 700,1)
    out_dim = (384, 350, 1)
    if downscale256:
        inp_dim = (256, 256, 1)
    if downscale256:
        out_dim = (256, 256, 1)
        
    inp = tf.keras.Input(shape=(x_length, *inp_dim))

    if architecture == 'Tian':
        x = encoder(inp, rnn_type, relu_alpha)
        output = decoder(x, rnn_type, relu_alpha)        
    elif architecture == 'AENN':
        output = generator_AENN(inp, rnn_type, relu_alpha, 
                                x_length, y_length, norm_method=norm_method, 
                                downscale256 = downscale256, batch_norm = batch_norm, num_filters = num_filters)
    else:
        raise Exception('Unkown architecture {}. Option are: Tian, AENN'.format(architecture))
    
    
    if not downscale256:
        # Apply mask to output
        # Mask pixels outside Netherlands  
        mask = tf.constant(get_mask_y(), 'float32')
        output = tf.keras.layers.Lambda(lambda x: x * mask, name='Mask')(output)
        if norm_method and norm_method == 'minmax_tanh':
            output = tf.keras.layers.subtract([output, 1-mask])  
    
    model = tf.keras.Model(inputs=inp, outputs=output, name='Generator')
    return model

def build_discriminator(relu_alpha, y_length, architecture = 'Tian', wgan = False, downscale256 = False, batch_norm = False,
                       drop_out = False):
    inp_dim = (384, 350, 1)
    if downscale256:
        inp_dim = (256, 256, 1)
    inp = tf.keras.Input(shape=(y_length, *inp_dim))
    
    if architecture == 'Tian':
        output = discriminator_Tian(inp, relu_alpha, wgan)
    elif architecture == 'AENN':
        output = discriminator_AENN(inp, relu_alpha, wgan, downscale256 = downscale256, 
                                    batch_norm = batch_norm, drop_out = drop_out)
    else:
        raise Exception('Unkown architecture {}. Option are: Tian, AENN'.format(architecture))
        
    model = tf.keras.Model(inputs=inp, outputs=output, name='Discriminator')
    return model

class GAN(tf.keras.Model):
    def __init__(self, inp_dim = (768,700,1), out_dim = (384, 350, 1), rnn_type='GRU', x_length=6, 
                 y_length=1, relu_alpha=0.2, architecture='Tian', l_adv = 1, l_rec = 0.01, g_cycles=1, 
                 label_smoothing = 0, norm_method = None, wgan = False, downscale256 = False, rec_with_mae=True,
                 batch_norm = False, drop_out = False, r_to_dbz = False):
        '''
        inp_dim: dimensions of input image(s), default 768x700
        out_dim: dimensions of the output image(s), default 384x350
        rnn_type: type of recurrent neural network can be LSTM or GRU
        x_length: length of input sequence
        y_length: length of output sequence
        relu_alpha: slope of leaky relu layers
        architecture: either 'Tian' or 'AENN'
        l_adv: weight of the adverserial loss for generator
        l_rec: weight of reconstruction loss (mse + mae) for the generator
        g_cycles: how many cycles to train the generator per train cycle
        label_smoothing: When > 0, we compute the loss between the predicted labels 
                          and a smoothed version of the true labels, where the smoothing 
                          squeezes the labels towards 0.5. Larger values of 
                          label_smoothing correspond to heavier smoothing
        norm_method: which normalization method was used. 
                     Can be none or minmax_tanh where data scaled to be between -1 and 1
        wgan: Option to use wasserstein loss (Not fully implemented yet)
        downscale256: if true than the images are downscaled to 256x256 by using bilinear interpolation
        rec_with_mae: if true the reconstruction loss is MSE+MAE if false, rec it consists of only the MSE
        batch_norm: if true batch normalization is applied after each convolution(/rnn) block
        drop_out: if true adds dropout layer after each conv block in the Discriminator (dropout rate of 0.2)
        r_to_dbz: If true the data values are in dbz not in r (mm/h)
        '''
        super(GAN, self).__init__()      

        self.generator = build_generator(rnn_type, x_length=x_length, 
                                         y_length = y_length, relu_alpha=relu_alpha, 
                                         architecture=architecture, norm_method=norm_method,
                                        downscale256 = downscale256, batch_norm = batch_norm)
        
        self.discriminator_frame = build_discriminator(y_length=1, 
                                                 relu_alpha=relu_alpha,
                                                architecture=architecture, wgan = wgan, 
                                                 downscale256 = downscale256, batch_norm = batch_norm, drop_out = drop_out)
        self.y_length = y_length
        if y_length > 1:
            self.discriminator_seq = build_discriminator(y_length=x_length+y_length, 
                                                     relu_alpha=relu_alpha,
                                                    architecture=architecture, wgan = wgan, 
                                                     downscale256 = downscale256, batch_norm = batch_norm, drop_out = drop_out)
        self.l_adv = l_adv
        self.l_rec = l_rec
        self.g_cycles=g_cycles
        self.label_smoothing=label_smoothing
        self.norm_method=norm_method
        self.r_to_dbz = r_to_dbz
        self.wgan = wgan
        self.rec_with_mae = rec_with_mae
        self.downscale256 = downscale256
        
    def compile(self, lr_g=0.0001, lr_d = 0.0001):
        super(GAN, self).compile()
        
        self.g_optimizer = Adam(learning_rate=lr_g) 
        self.d_optimizer = Adam(learning_rate=lr_d)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.loss_fn_d = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        self.loss_mae = tf.keras.losses.MeanAbsoluteError()
        
        self.g_loss_metric_frame = tf.keras.metrics.Mean(name="g_loss_frame")
        self.g_loss_metric_seq = tf.keras.metrics.Mean(name="g_loss_seq")
        
        self.d_loss_metric_frame= tf.keras.metrics.Mean(name="d_loss_frame")
        self.d_loss_metric_seq = tf.keras.metrics.Mean(name="d_loss_seq")
        
        self.d_acc_frame = tf.keras.metrics.BinaryAccuracy(name='d_acc_frame')
        self.d_acc_seq = tf.keras.metrics.BinaryAccuracy(name='d_acc_seq')
        
        self.rec_metric = tf.keras.metrics.Mean(name="rec_loss")

        if self.wgan:
            self.opt = RMSprop(lr=0.00005)
            self.loss_fn = wasserstein_loss
            
    def loss_rec(self, target, pred, MAE = True):
        '''
        Reconstruction loss: sum of MSE and MAE.
        mae: If false the reconstruction loss is equal to the MSE, this was found to perform better
        '''
        g_loss_mse = self.loss_mse(target, pred)
        if MAE:
            g_loss_mae = self.loss_mae(target, pred)       
        else:
            g_loss_mae = 0
        return g_loss_mse + g_loss_mae
    
    def call(self, x):
        """Run the model."""
        y_pred = self.generator(x)
        return y_pred

    @property
    def metrics(self):
        return [self.d_loss_metric_frame, self.d_loss_metric_seq, 
                self.g_loss_metric_frame, self.g_loss_metric_seq, 
                self.rec_metric, self.d_acc_frame, self.d_acc_seq]
    
    def train_disc_seq(self, inp, labels, train = True ):
        if train:
            with tf.GradientTape() as tape:
                predictions = self.discriminator_seq(inp)
                d_loss_seq = self.loss_fn_d(labels, predictions)
            grads = tape.gradient(d_loss_seq, self.discriminator_seq.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator_seq.trainable_weights)
            )
        else:
            predictions = self.discriminator_seq(inp)
            d_loss_seq = self.loss_fn_d(labels, predictions)
        # Update D accuracy metric
        self.d_acc_seq.update_state(labels, predictions)
        return d_loss_seq
    
    def train_disc_frame(self, inp, labels, train = True ):
        if train:
            with tf.GradientTape() as tape:
                d_loss_frame = 0
                for i in range(self.y_length):
                    frame = inp[:,i:i+1]
                    predictions = self.discriminator_frame(frame)
                    d_loss_frame += self.loss_fn_d(labels, predictions)
            grads = tape.gradient(d_loss_frame, self.discriminator_frame.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator_frame.trainable_weights)
            )
        else:
            d_loss_frame = 0
            for i in range(self.y_length):
                frame = inp[:,i:i+1]
                predictions = self.discriminator_frame(frame)
                d_loss_frame += self.loss_fn_d(labels, predictions)
                
        # Update D accuracy metric
        # TODO: calculate average accuracy over the frames
        # Now d_acc_frame is accuracy on the last frame
        self.d_acc_frame.update_state(labels, predictions)
        return d_loss_frame
    
    def train_discriminators(self, xs , ys, batch_size, train = True):
        # Decode them to fake images
        generated_images = self.generator(xs)

        # Combine them with real images
        combined_images = tf.concat([generated_images, ys], axis=0)
        
        # concatenate input and predictions in feature dimensions
        # D then looks at the whole sequence (cGAN)
        seq_pred = tf.concat([xs, generated_images], axis=1)
        seq_real = tf.concat([xs, ys], axis=1)
        combined_sequences = tf.concat([seq_pred, seq_real], axis=0)
        
        # Assemble labels discriminating fake from real images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Train the frame discriminator
        d_loss_frame = self.train_disc_frame(combined_images, labels, train)
        
        # Train the sequence discriminator
        if self.y_length > 1:
            d_loss_seq = self.train_disc_seq(combined_sequences, labels, train)
        else:
            d_loss_seq = d_loss_frame
        return d_loss_frame, d_loss_seq

    def train_generator(self, xs, ys, batch_size, train = True):
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        if train:
            for _ in range(self.g_cycles):
                with tf.GradientTape() as tape:
                    generated_images = self.generator(xs)
                    adv_loss_frame = self.train_disc_frame(generated_images, misleading_labels, train = False)
                    if self.y_length > 1:
                        seq_pred = tf.concat([xs, generated_images], axis=1)
                        adv_loss_seq = self.train_disc_seq(seq_pred, misleading_labels, train = False)
                    else:
                        adv_loss_seq = adv_loss_frame
                    g_loss_adv = adv_loss_frame + adv_loss_seq
                    g_loss_rec = self.loss_rec(ys, generated_images, self.rec_with_mae)
                    g_loss =  self.l_adv * g_loss_adv  + self.l_rec * g_loss_rec           
                grads = tape.gradient(g_loss, self.generator.trainable_weights)
                self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        else:
            generated_images = self.generator(xs)
            adv_loss_frame = self.train_disc_frame(generated_images, misleading_labels, train = False)
            if self.y_length > 1:
                seq_pred = tf.concat([xs, generated_images], axis=1)
                adv_loss_seq = self.train_disc_seq(seq_pred, misleading_labels, train = False)
            else:
                adv_loss_seq = 0
            g_loss_adv = adv_loss_frame + adv_loss_seq
            g_loss_rec = self.loss_rec(ys, generated_images, self.rec_with_mae)
            g_loss =  self.l_adv * g_loss_adv  + self.l_rec * g_loss_rec 
        return adv_loss_frame, adv_loss_seq, g_loss_rec
    
    def undo_prep(self, x):
        x = batchcreator.undo_prep(x, norm_method=self.norm_method, r_to_dbz=self.r_to_dbz, downscale256=self.downscale256)
        return x
    
    def model_step(self, batch, train = True):
        '''
        This function performs train_step
        batch: batch of x and y data
        train: wether to train the model
               True for train_step, False when performing test_step
        '''
        xs, ys = batch
        batch_size = tf.shape(xs)[0]

        d_loss_frame, d_loss_seq = self.train_discriminators(xs,ys,batch_size,train)
        g_loss_frame, g_loss_seq, g_loss_rec  = self.train_generator(xs,ys,batch_size,train)


        # Update metrics
        self.d_loss_metric_frame.update_state(d_loss_frame)
        self.d_loss_metric_seq.update_state(d_loss_seq)
        self.g_loss_metric_frame.update_state(g_loss_frame)
        self.g_loss_metric_seq.update_state(g_loss_seq)
        self.rec_metric.update_state(g_loss_rec)
         
        if self.y_length > 1:    
            return {
                "d_loss_frame": self.d_loss_metric_frame.result(),
                "d_loss_seq": self.d_loss_metric_seq.result(),
                "g_loss_frame": self.g_loss_metric_frame.result(),
                "g_loss_seq": self.g_loss_metric_seq.result(),
                "rec_loss": self.rec_metric.result(),
                "d_acc_frame": self.d_acc_frame.result(),
                'd_acc_seq': self.d_acc_seq.result()
            } 
        else:
            return {
            "d_loss_frame": self.d_loss_metric_frame.result(),
            "g_loss_frame": self.g_loss_metric_frame.result(),
            "rec_loss": self.rec_metric.result(),
            "d_acc_frame": self.d_acc_frame.result(),
        } 
            
    def train_step(self, batch):
        metric_dict = self.model_step(batch, train = True)       
        return metric_dict
    
    def test_step(self, batch):
        metric_dict = self.model_step(batch, train = False)       
        return metric_dict