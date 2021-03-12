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
    
    path_mask = '/usr/people/schreurs/KNMI_Internship_GANs/precipitation_forecasting/mask.npy'

    if os.path.isfile(path_mask):
        mask = np.load(path_mask)
    else:
        # Get the mask for the input data
        y_path = '/nobackup_1/users/schreurs/project_GAN/dataset_aart'
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
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator = build_discriminator()
        self.generator = build_generator()
 
    
    def compile(self, optimizer='adam'):
        super(GAN, self).compile()
        
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)   
        
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
    
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
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
