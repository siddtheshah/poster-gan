import urllib.request
import zipfile

DOWNLOAD_PATH = '/tmp/movie_lens_data.zip'

url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
urllib.request.urlretrieve(url, DOWNLOAD_PATH)

with zipfile.ZipFile(DOWNLOAD_PATH) as zip_ref:
    zip_ref.extractall('/tmp/movie_lens_data')