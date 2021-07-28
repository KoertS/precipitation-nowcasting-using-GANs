import tensorflow as tf
import wandb
import numpy as np
from plotter import plot_target_pred
import matplotlib.pyplot as plt 
from batchcreator import minmax, dbz_to_r, DataGenerator
import batchcreator

class ImageLogger(tf.keras.callbacks.Callback):
    '''
    This visualizes the predictions of the model.
    generator: datagenerator that yields x and y pairs
    persistent: when false a new random batch of images is plotted after each epoch. 
                if true the same random batch of images is drawn after each epoch
    train_data: True if generator is the training generator, false if it is the validation generator
    '''
    def __init__(self, generator, persistent = False, train_data = True):
        self.generator = generator
        
        self.persistent = persistent
        if persistent:
            self.idx = np.random.randint(0,len(self.generator))
            
        # Set the name displayed in the wandb interface
        if train_data:
            self.wandb_title = "Targets & Predictions"
        else:
            self.wandb_title = "Targets & Predictions (val)"
        super(ImageLogger, self).__init__()

    def on_epoch_end(self, logs, epoch):
        if self.persistent:
            idx = self.idx
        else:
            idx = np.random.randint(0,len(self.generator))
          
        xs, ys = self.generator.__getitem__(idx)
       
        predictions = self.model.predict(xs)
      
        # Undo the preprocessing
        # Convert back to mm/h
        predictions = batchcreator.undo_prep(predictions, norm_method = self.generator.norm_method, 
                                             r_to_dbz = self.generator.convert_to_dbz, downscale256 = self.generator.downscale256)
        ys = batchcreator.undo_prep(ys, norm_method = self.generator.norm_method, 
                                             r_to_dbz = self.generator.convert_to_dbz, downscale256 = self.generator.downscale256)
        
        plots = []
        for i in range(len(ys)):
            plot = plot_target_pred(ys[i], predictions[i])
            plots.append(plot)
            
        wandb.log({self.wandb_title: [wandb.Image(plot)
                              for plot in plots]})
        plt.close('all')

class ValidateLogger(tf.keras.callbacks.Callback):
    '''
    The validate logger calculates the mse of the model on the validation set.
    The mse is calculated on the original data. The model predictions are upscaled back to 765, 700
    The predictions are unnormalized and converted back to mm/h.
    The two generators are not shuffled so if iterate over them they will sync up
    '''
    def __init__(self, generator):
        self.generator = generator
        
        # Copy the validation generator but change it so that it loads the unpreprocessed data 
        self.generator_unprep = DataGenerator(generator.list_IDs, batch_size=generator.batch_size, x_seq_size=generator.inp_shape[0], 
                                       y_seq_size=generator.out_shape[0], norm_method=None, load_prep=False, downscale256 = False, 
                                              convert_to_dbz = False, y_is_rtcor = generator.y_is_rtcor, shuffle=False, crop_y=False)
        
        self.epoch_freq = 10
        
        self.mse = tf.keras.losses.MeanSquaredError()
        super(ValidateLogger, self).__init__()
    
    def validate_mse():
        for (xs_prep, ys_prep), (_, ys) in zip(self.generator, self.generator_unprep):
            # 0.01mm/h -> 1mm/h
            ys = ys/100
            # Undo normalization and convert dbz to r (mm/h)
            ys_pred = self.model.undo_prep(self.model(xs))
            # Upsample the image using bilinear interpolation
            ys_pred =  tf.convert_to_tensor([tf.image.resize(y, (768, 768)) for y in y_pred])
            # Original shape was 765x700, crop prediction so that it fits this
            ys_pred = ys_pred[:,:,:-3, :-68]
            mse = self.mse(y_pred, ys).numpy()               
        wandb.log({'val_mse_mm/h': mse})    
        
    def on_epoch_end(self, logs, epoch):
        # Validate every n epochs and at the last epoch
        if epoch % self.epoch_freq == 0 or epoch == self.params.get('epochs', -1):
            validate_mse()
        
            
        
class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        self.generator = generator

        super(GradientLogger, self).__init__()
    
    def on_epoch_end(self, logs, epoch):
        xs, ys = self.generator.__getitem__(np.random.randint(0,len(self.generator)))
        
        batch_size = tf.shape(xs)[0]


        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            generated_images = self.model.generator(xs)
            adv_loss_frame = self.model.train_disc_frame(generated_images, misleading_labels, train = False)
            
            if self.model.y_length > 1:
                seq_pred = tf.concat([xs, generated_images], axis=1)
                adv_loss_seq = self.model.train_disc_seq(seq_pred, misleading_labels, train = False)
            else:
                adv_loss_seq = adv_loss_frame
            g_loss_adv = adv_loss_frame + adv_loss_seq
            g_loss_rec = self.model.loss_rec(ys, generated_images, self.model.rec_with_mae)
            g_loss =  self.model.l_adv * g_loss_adv  + self.model.l_rec * g_loss_rec           
        grads_g = tape.gradient(g_loss, self.model.generator.trainable_weights)
  
        grads_g = [item.numpy().flatten() for sublist in grads_g for item in sublist]
        grads_g = [item for sublist in grads_g for item in sublist]
            
        wandb.log({'grads_g': wandb.Histogram(grads_g)})        