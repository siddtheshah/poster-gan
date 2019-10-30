#! /usr/bin/python3

import pandas as pd
import wget
import os

IMAGE_FOLDER = "/tmp/movie_poster_images"

frame = pd.read_csv('MovieGenre.csv', encoding="ISO-8859-1")

for movieId, url in zip(frame['imdbId'], frame['Poster']):
    # print(os.path.join(IMAGE_FOLDER, str(movieId) + ".jpg"))
    wget.download(url, os.path.join(IMAGE_FOLDER, str(movieId) + ".jpg"))