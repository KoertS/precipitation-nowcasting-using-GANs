import tensorflow as tf
import wandb
import numpy as np

class ImageLogger(tf.keras.callbacks.Callback):
    '''
    This visualizes the predictions of the model.
    x_test: model input (validation)
    y_test: target data
    n: indicates how many images to plot
    '''
    def __init__(self, x_test, y_test, n=4):
        self.n = n
        self.x_test = x_test
        self.y_test = y_test

        super(ImageLogger, self).__init__()

    def on_epoch_end(self, logs, epoch):
        indexes = np.random.randint(0, len(self.x_test), self.n)

        sample_images = self.x_test[indexes]
        sample_targets = self.y_test[indexes]

        images = []
        
        reconstructions = self.model.predict(sample_images)[:self.n]
            
        for i in range(self.n):
            images.append(sample_targets[i].reshape(384,350))
        reconstructions = [reconstruction.reshape(384,350) for reconstruction in reconstructions]

        wandb.log({"images": [wandb.Image(image)
                              for image in images],
                   "epoch": epoch,
                  })
        wandb.log({"reconstructions": [wandb.Image(reconstruction)
                              for reconstruction in reconstructions],
                  "epoch": epoch
                  })

class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, n=4):
        self.n = n
        self.x_test = x_test
        self.y_test = y_test

        super(GradientLogger, self).__init__()

    
    def on_epoch_end(self, logs, epoch):
        indexes = np.random.randint(0, len(self.x_test), self.n)

        xs = self.x_test[indexes]
        ys = self.y_test[indexes]
        
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
            predictions = self.model.discriminator(self.model.generator(xs))
            g_loss_gan = self.model.loss_fn(misleading_labels, predictions)
     
           # g_loss_mse = self.loss_mse(ys, predictions)
            g_loss = g_loss_gan #+ g_loss_mse
        
        grads_g = tape.gradient(g_loss, self.model.generator.trainable_weights)
        
        grads_g = [item.numpy().flatten() for sublist in grads_g for item in sublist]
        grads_g = [item for sublist in grads_g for item in sublist]
        
        grads_d = [item.numpy().flatten() for sublist in grads_d for item in sublist]
        grads_d = [item for sublist in grads_d for item in sublist]
            
        wandb.log({'grads_g': wandb.Histogram(grads_g), 'grads_d': wandb.Histogram(grads_d)})
        
        