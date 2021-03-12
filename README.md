# Nowcasting Precipitation using GANs

## GANs on MNIST
In [this folder](https://github.com/KoertS/KNMI_Internship_GANs/tree/main/dcgan_MNIST) I apply a conditional GAN to a hand written number dataset in order to gain some more experience with GANs and cGANs.

## GANs on Precipitation Data

[Here](https://github.com/KoertS/KNMI_Internship_GANs/tree/main/precipitation_forecasting) I work with the precipitation radar data. The real time (rtcor) data is matched with the Overreem data to create input output pairs. This is done in the [batchcreator](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/batchcreator.py) module. 
The [model_builder](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/model_builder.py) is used to create a model. Currently I implemented a GAN model that is based the paper from [Tian et al.](https://ieeexplore.ieee.org/abstract/document/8777193) on precipitation nowcasting. 
In [this](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/testrun.ipynb) notebook I perform a test run to get an idea of the training time.

## Paper
Work in progress of the paper I'm writing can be found in [overleaf](https://www.overleaf.com/read/nqbdxkjnnqyv).
