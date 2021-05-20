import h5py
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import netCDF4
from pyproj import Proj, transform
from mpl_toolkits.basemap import Basemap
import config

def plot_on_map(rdr, ftype='.nc', res='l',colorbar=True):
    '''
    Plot radar file on top of map.
    rdr: image file
    ftype: file extension, can be either '.nc' or '.h5'
    res: resolution, can be c (crude), l (low), i (intermediate), h (high), f (full) 
    '''
    
    dir_aart = config.dir_aart
    aart_fbase = 'RAD_NL25_RAC_MFBS_EM_5min_'
    proj = Proj("+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0")
    
    if ftype != '.nc' and ftype !='.h5':
        print('Error: invalid file extension {}, should be .nc or .h5'.format(ftype))
    # All images are plotted on the same map
    # Get the map from random nc file
    path = dir_aart + aart_fbase + '201901010000.nc'
    with netCDF4.Dataset(path, 'r') as ds:
        # Get coordinates of the pixels
        xx, yy = np.meshgrid(ds['x'][:], ds['y'][:])
        lon, lat = proj(xx, yy, inverse=True)
        # Plot values on map
        iso_dict = ds['iso_dataset'].__dict__
        min_x = float(iso_dict['min_x'].replace('f', ''))
        min_y = float(iso_dict['min_y'].replace('f', ''))
        max_x = float(iso_dict['max_x'].replace('f', ''))
        max_y = float(iso_dict['max_y'].replace('f', ''))
        
        if ftype == '.nc':
            rain = ds['image1_image_data'][:]
            mask = rain.mask
    
    if  ftype == '.h5':
        dir_rtcor = config.dir_rtcor
        rtcor_fbase = 'RAD_NL25_RAC_RT_'
        path = dir_rtcor + rtcor_fbase + '201901010000.h5'
        with h5py.File(path, 'r') as f:
                rain = f['image1']['image_data'][:]
                mask = (rain == 65535)

    
    # Mask the data
    mx = np.ma.masked_array(rdr, mask)
    # Plot the precipitation on map  
    mp = Basemap(projection='stere',
                         lat_0=90,
                         lon_0=0, 
                         lat_ts=60,   
                         llcrnrlon=min_x,   # lower longitude
                         llcrnrlat=min_y,    # lower latitude
                         urcrnrlon=max_x,   # uppper longitude
                         urcrnrlat=max_y,   # uppper latitude
                         resolution=res
                        )
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    xx, yy = mp(lon, lat)

    vmin=-0.00001    
    cmap = cm.viridis
    plt.imshow(mx, vmin = vmin, cmap=cmap, origin='upper', 
           extent=[xx.min(), xx.max(), yy.min(), yy.max()])
    if colorbar:
        plt.colorbar()