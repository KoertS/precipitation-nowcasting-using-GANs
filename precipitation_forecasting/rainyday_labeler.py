import numpy as np
from skimage import morphology
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import h5py

radar_dir = '/ceph/csedu-scratch/project/kschreurs/dataset_radar/'
label_dir = '/ceph/csedu-scratch/project/kschreurs/rtcor_rain_labels/'


files = sorted([f for f in listdir(radar_dir) if isfile(join(radar_dir, f)) and f.startswith('2019')])

def is_rainy(rdr):
    cluttermask = ~np.load('cluttermask.npy')
    
    # Calculate gradien magnitudes
    vgrad= np.gradient(rdr)
    mag = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
    
    # Ignore pixels that tend to contain clutter
    rdr_clean = rdr * cluttermask
    # Ignore rainy objects smaller than 15
    cleaned = morphology.remove_small_objects(rdr>0, min_size=9, connectivity=8)   
    rdr_clean = rdr_clean*cleaned

    # Label as rainy
    # If not many high gradients (clutter) and total precipitation above 30mm
    if len(np.argwhere(mag>500)) < 130 and np.sum(rdr_clean) > 3000:
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
        except:
            print("Error: could not read image1 data, file {}".format(path))
    return radar_img
    
def make_dir(dir_name):
  '''
  Create directory if it does not exist
  '''
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    
make_dir(label_dir)
for f in tqdm(files):
    try:
        rdr = load_h5(radar_dir+f)
        rainy = is_rainy(rdr)
    except:
        rainy = False
        
    label_fn = label_dir + f
    np.save(label_fn, rainy) 