#! /usr/bin/python3

import pandas as pd
import wget
import os

IMAGE_FOLDER = "movie_poster_images"

frame = pd.read_csv('MovieGenre.csv', encoding="ISO-8859-1")

errors = []

for movieId, url in zip(frame['imdbId'], frame['Poster']):
    try:
        wget.download(url, os.path.join(IMAGE_FOLDER, str(movieId) + ".jpg"))
    except:
        errors.append(movieId)

print("Errors: ", errors)