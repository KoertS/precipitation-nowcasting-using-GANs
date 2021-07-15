import numpy as np
from skimage import morphology
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import h5py
import config

import sys

# Get full command-line arguments
full_cmd_arguments = sys.argv

year = 0
overwrite = False

if len(full_cmd_arguments) >= 2:
    year = full_cmd_arguments[1]
    print('Computing labels for the year {}'.format(year))
    if len(full_cmd_arguments) >= 3:
        o_arg = full_cmd_arguments[2]
        
        if o_arg == '-o':
            print('Overwriting previous labels')
            overwrite = True
        else:
            print('Invalid second argument. Expected -o (overwrite)')
            exit()
else:
    print('Missing argument year (example: rainyday_labeler.py 2019)')
    exit()

radar_dir =  config.dir_rtcor
label_dir = config.dir_labels_heavy

root = radar_dir + year
files = sorted([name for path, subdirs, files in os.walk(root) for name in files])

#cluttermask = ~np.load('cluttermask.npy')
   
path = config.dir_rtcor +  '2019/01/{}201901010000.h5'.format(config.prefix_rtcor)
with h5py.File(path, 'r') as f:
    rain = f['image1']['image_data'][:]
    mask = (rain == 65535)
nr_unmasked_pixels = (765*700)-np.sum(mask)#+~cluttermask)  

def has_clutter(rdr):
    # Calculate gradien magnitudes 
    gx, gy = np.gradient(rdr)        
    grad = np.hypot(gx,gy)
    # Pixel is seen as abnormal if gradient is higher than 30
    abnormal_pixels = np.sum(grad>30)
    # Image is discared if it has more than 50 abnormal pixels
    return abnormal_pixels > 50

def is_rainy(rdr):    
    # Label as rainy
    # If not many high gradients (clutter) and avg > 0.01mm
    avg_rain = np.sum(rdr)/nr_unmasked_pixels
    
    clutter = has_clutter(rdr)
    if not clutter and avg_rain > 0.01:
        return True
    return False

def load_h5(path):
    '''
    The orginial input images are stored in .h5 files. 
    This function loads them and converts them to numpy arrays
    '''
    radar_img = None
    with h5py.File(path, 'r') as f:
        try:
            radar_img = f['image1']['image_data'][:]

            ## Set pixels out of image to 0
            out_of_image = f['image1']['calibration'].attrs['calibration_out_of_image']
            radar_img[radar_img == out_of_image] = 0
            # Sometimes 255 or other number (244) is used for the calibration
            # for out of image values, so also check the first pixel
            radar_img[radar_img == radar_img[0][0]] = 0
            
            # Convert the values to mm/h (from 0.01mm/5min)
            radar_img = (radar_img/100)*12
            radar_img = np.clip(radar_img, 0,100)
        except:
            print("Error: could not read image1 data, file {}".format(path))
    return radar_img
    
def make_dir(dir_name):
  '''
  Create directory if it does not exist
  '''
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
# make directories
make_dir(label_dir)
make_dir(label_dir+year)
for m in range (1,13):
    make_dir(label_dir+year+'/{:02d}'.format(m))

for f in tqdm(files):
    ts = f.replace(config.prefix_rtcor, '')
    ts = ts.replace('.h5', '')

    year = ts[:4]
    month = ts[4:6]

    label_fn = label_dir + '{Y}/{m}/{ts}.npy'.format(Y=year, m=month, ts=ts)

    if not os.path.isfile(label_fn) or overwrite:
        try:
            rdr = load_h5(radar_dir+'/{}/{}/{}'.format(year,month,f))
            rainy = is_rainy(rdr)
        except Exception as e:
            rainy = False
            print(e)
        np.save(label_fn, rainy)

