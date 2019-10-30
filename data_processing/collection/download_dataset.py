#! /usr/bin/python

import pandas as pd
import urllib
import os

IMAGE_FOLDER = "/tmp/movie_poster_images"

frame = pd.read_csv('MovieGenre.csv', encoding="ISO-8859-1")

for movieId, url in zip(frame['imdbId'], frame['Poster']):
    # print(os.path.join(IMAGE_FOLDER, str(movieId) + ".jpg"))
    urllib.urlretrieve(url, os.path.join(IMAGE_FOLDER, str(movieId) + ".jpg"))