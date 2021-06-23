import tensorflow as tf
import wandb
import numpy as np
from radarplot import plot_target_pred
import matplotlib.pyplot as plt 
from batchcreator import minmax

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
      
        if self.model.norm_method and self.model.norm_method == 'minmax_tanh':
            predictions = minmax(predictions, tanh=True, undo=True)   
            ys = minmax(ys, tanh = True, undo=True)  
           
        plots = []
        for i in range(len(ys)):
            plot = plot_target_pred(ys[i], predictions[i])
            plots.append(plot)
            
        wandb.log({self.wandb_title: [wandb.Image(plot)
                              for plot in plots]})
        plt.close('all')

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
                adv_loss_seq = self.model.train_disc_seq(generated_images, misleading_labels, train = False)
            else:
                adv_loss_seq = adv_loss_frame
            g_loss_adv = adv_loss_frame + adv_loss_seq
            g_loss_rec = self.model.loss_rec(ys, generated_images, self.model.rec_with_mae)
            g_loss =  self.model.l_adv * g_loss_adv  + self.model.l_rec * g_loss_rec           
        grads_g = tape.gradient(g_loss, self.model.generator.trainable_weights)
  
        grads_g = [item.numpy().flatten() for sublist in grads_g for item in sublist]
        grads_g = [item for sublist in grads_g for item in sublist]
            
        wandb.log({'grads_g': wandb.Histogram(grads_g)})        