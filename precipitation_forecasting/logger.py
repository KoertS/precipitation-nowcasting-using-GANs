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
    reconstructions = []
    
    for i in range(self.n):
        reconstruction = self.model.predict(sample_images[i])

        images.append(sample_targets[i].reshape(384,350))
        reconstructions.append(reconstruction.reshape(384,350))

    wandb.log({"images": [wandb.Image(image)
                          for image in images],
               "epoch": epoch
              })
    wandb.log({"reconstructions": [wandb.Image(reconstruction)
                          for reconstruction in reconstructions],
              "epoch": epoch
              })