# Nowcasting Precipitation using GANs

This repository contains the code and models to replicate the experiments in my [thesis](https://www.ru.nl/publish/pages/769526/koert_schreurs.pdf) that I worked on during my internship at KNMI.

## Data
I used real-time precipitation radar data. The data past 2018 is publicly available at the [KNMI data platform](https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m_tar/versions/1.0/files).

Additionally, we also explored the use of machine learning for forecasting radar data that was further bias-corrected. A archive of this radar product can be found [here](https://dataplatform.knmi.nl/dataset/rad-nl25-rac-mfbs-em-5min-netcdf4-2-0)

## GANs on MNIST
In [this folder](https://github.com/KoertS/KNMI_Internship_GANs/tree/main/dcgan_MNIST) I apply a conditional GAN to a hand written number dataset in order to gain some more experience with GANs and cGANs.

Additionally, we also explored the use of machine learning for forecasting radar data that was further bias-corrected. An archive of this radar product can be found [here]
## GANs on Precipitation Data

[Here](https://github.com/KoertS/KNMI_Internship_GANs/tree/main/precipitation_forecasting) I work with the precipitation radar data. The real time (rtcor) data is matched with the Overreem data to create input output pairs. This is done in the [batchcreator](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/batchcreator.py) module. 
The [model_builder](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/model_builder.py) is used to create a model. Currently I implemented a GAN model that is based the paper from [Tian et al.](https://ieeexplore.ieee.org/abstract/document/8777193) on precipitation nowcasting. 
In [this](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/testrun.ipynb) notebook I perform a test run to get an idea of the training time.
