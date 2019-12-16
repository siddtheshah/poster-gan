#! /usr/bin/python3

import pandas as pd
import wget
import os
from PIL import Image

IMAGE_FOLDER = "I:\ml_storage\movie_poster_images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

frame = pd.read_csv('MovieGenre.csv', encoding="ISO-8859-1")

errors = []

for movieId, url in zip(frame['imdbId'], frame['Poster']):
    try:
        image_path = os.path.join(IMAGE_FOLDER, str(movieId) + ".jpg")
        wget.download(url, image_path)
        img = Image.open(image_path)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(image_path)
    except:
        errors.append(movieId)

print("Errors: ", errors)