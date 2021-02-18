import urllib.request
import os
from tqdm import tqdm

# https://github.com/tqdm/tqdm#hooks-and-callbacks
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
  '''
  Download file in url. Output is written to output_path.
  Shows download progressbar
  '''
  with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
    urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def make_dir(dir_name):
  '''
  Create directory if it does not exist
  '''
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Replace 'YEAR' with a year to get download url of the given year
base_url = 'http://opendap.knmi.nl/knmi/thredds/fileServer/radarprecipclim/RAD_NL25_RAC_MFBS_EM_5min_NC/RAD_NL25_RAC_MFBS_EM_5min_YEAR_NETCDF.zip'
output_dir='/nobackup/users/schreurs/project_GAN/dataset_aart'
make_dir(output_dir)

# Create list of year strings
years = [str(year) for year in range(2008,2021,1)]
for year in years:
  year_url = base_url.replace('YEAR', year)

  # Define output path
  year_dir = output_dir + '/{}'.format(year)
  make_dir(year_dir)
  output_path = year_dir + '/' + '{}.zip'.format(year)
  
  download_url(year_url, output_path)
