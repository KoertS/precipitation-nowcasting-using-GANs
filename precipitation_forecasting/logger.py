import tensorflow as tf
import wandb
import numpy as np
from radarplot import plot_target_pred
import matplotlib.pyplot as plt 
from batchcreator import minmax

class ImageLogger(tf.keras.callbacks.Callback):
    '''
    This visualizes the predictions of the model.
    x_test: model input (validation)
    y_test: target data
    n: indicates how many images to plot
    '''
    def __init__(self, generator):
        self.generator = generator
        super(ImageLogger, self).__init__()

    def on_epoch_end(self, logs, epoch):
        xs, ys = self.generator.__getitem__(np.random.randint(0,len(self.generator)))
       
        predictions = self.model.predict(xs)
        images = ys
        
        if self.model.norm_method and self.model.norm_method == 'minmax_tanh':
            predictions = minmax(predictions, tanh=True, undo=True)   
            images = minmax(images, tanh = True, undo=True)  
           
        predictions = np.squeeze(predictions)
        images =np.squeeze(ys)
        plots = []
        
 
            
        for i in range(len(images)):
            plot = plot_target_pred(images[i], predictions[i])
            plots.append(plot)
            
        wandb.log({"Targets & Predictions": [wandb.Image(plot)
                              for plot in plots]})
        plt.close('all')

class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        self.generator = generator

        super(GradientLogger, self).__init__()

    
    def on_epoch_end(self, logs, epoch):
        xs, ys = self.generator.__getitem__(np.random.randint(0,len(self.generator)))
        
        batch_size = tf.shape(xs)[0]

        # Decode them to fake images
        generated_images = self.model.generator(xs)

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
            predictions = self.model.discriminator(combined_images)
            d_loss = self.model.loss_fn(labels, predictions)
        grads_d = tape.gradient(d_loss, self.model.discriminator.trainable_weights)

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated_images = self.model.generator(xs)
            predictions = self.model.discriminator(generated_images)
            g_loss_gan = self.model.loss_fn(misleading_labels, predictions)
            g_loss_mse = self.loss_mse(ys, generated_images)
            g_loss_mae = self.loss_mae(ys, generated_images)
            g_loss = self.l_g * g_loss_gan  + self.l_rec * (g_loss_mse+g_loss_mae)       
        grads_g = tape.gradient(g_loss, self.model.generator.trainable_weights)
        
        grads_g = [item.numpy().flatten() for sublist in grads_g for item in sublist]
        grads_g = [item for sublist in grads_g for item in sublist]
        
        grads_d = [item.numpy().flatten() for sublist in grads_d for item in sublist]
        grads_d = [item for sublist in grads_d for item in sublist]
            
        wandb.log({'grads_g': wandb.Histogram(grads_g), 'grads_d': wandb.Histogram(grads_d)})        