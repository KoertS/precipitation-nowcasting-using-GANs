# Nowcasting Precipitation using GANs

This repository contains the code and models to replicate the experiments in my [thesis](https://www.ru.nl/publish/pages/769526/koert_schreurs.pdf) that I worked on during my internship at KNMI.

# Running the code

In order to run the code on your own machine, you need to specify a few things

## Data
I used real-time precipitation radar data from 2008 till 2020. The data past 2018 is publicly available at the [KNMI data platform](https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m_tar/versions/1.0/files). This dataset was used to train and validate the model

Additionally, we also explored the use of machine learning for forecasting radar data that was further bias-corrected. An archive of this radar product can be found [here](https://dataplatform.knmi.nl/dataset/rad-nl25-rac-mfbs-em-5min-netcdf4-2-0). Note: this dataset was not used to train the model

A subselection of the data was used, only samples with sufficient rain were included. Each sample in the dataset was labeled as rainy or not rainy. 
To obtain these labels you can run the python script [rainyday_labeler.py](https://github.com/KoertS/precipitation-nowcasting-using-GANs/blob/main/precipitation_forecasting/rainyday_labeler.py), with as argument the year you want to label (example: rainyday_labeler.py 2019, this would label all the samples from 2019)

A generator is used to retrieve parts of the data during runtime (Datagenerator class in [batchcreator module](https://github.com/KoertS/precipitation-nowcasting-using-GANs/blob/main/precipitation_forecasting/batchcreator.py)). The generator loads the input, target pairs by filename. To create these input and target pairs you can run the python script [create_traindata_IDs](https://github.com/KoertS/precipitation-nowcasting-using-GANs/blob/main/precipitation_forecasting/create_traindata_IDs.py) and change the time interval and input and output length and the filename to your needs.
 
In the paper the samples were preprocessed. This can be done by running the rtcor2npy function in the [preprocess_data_rtcor notebook](https://github.com/KoertS/precipitation-nowcasting-using-GANs/blob/main/precipitation_forecasting/preprocess_data_rtcor.ipynb) with parameter preprocess set to true. 
As input to the rtcor2npy function receives a list of filenames to preprocess. It saves a lot of computation time if you only preproces the rainy samples. So you probably want to create a list of samples labeled with rainy (both the input and target samples):
```python
# Load all target files in the training set
fn_rtcor_input = np.load('datasets/FILENAME_DATASET.npy', allow_pickle = True)[:,0]
fn_rtcor_target = np.load('datasets/FILENAME_DATASET.npy', allow_pickle = True)[:,1]

filenames_rtcor= np.append(fn_rtcor_input, fn_rtcor_target)
# flatten the array
filenames_rtcor = [item for sublist in filenames_rtcor for item in sublist]
len(filenames_rtcor)

rtcor2npy(config.dir_rtcor, config.dir_rtcor_prep, overwrite = False, preprocess = True, filenames = filenames_rtcor)
```


## Config

In the [config.py](https://github.com/KoertS/precipitation-nowcasting-using-GANs/blob/main/precipitation_forecasting/config.py) change the path to your data (path_data) and to your project (path_project) to match your system. Furthermore the real-time dataset can have different names (rtcor or RAC). Check your data to see if the prefix of your data matches the one stated in the config file (prefix_rtcor).


# GANs on Precipitation Data

[Here](https://github.com/KoertS/KNMI_Internship_GANs/tree/main/precipitation_forecasting) I work with the precipitation radar data. The real time (rtcor) data is matched with the Overreem data to create input output pairs. This is done in the [batchcreator](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/batchcreator.py) module. 
The [model_builder](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/model_builder.py) is used to create a model. Currently I implemented a GAN model that is based the paper from [Tian et al.](https://ieeexplore.ieee.org/abstract/document/8777193) on precipitation nowcasting. 
In [this](https://github.com/KoertS/KNMI_Internship_GANs/blob/main/precipitation_forecasting/testrun.ipynb) notebook I perform a test run to get an idea of the training time.
