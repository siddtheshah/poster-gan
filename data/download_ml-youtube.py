import pandas as pd
from pytube import YouTube
import sys
import os

#import youtube_dl as ydl

def make_directories(video_names):
  if not os.path.exists('../../ml_trailers'):
    os.mkdir('../../ml_trailers')

def download_videos(video_names, movie_imdb_id_mapping):
  fail_download = []

  for index, row in video_names.iterrows():
    print("Video number: " + str(index))

    youtube_id = row[0]
    movie_id = row[1]
    title = row[2]

    if ', The' in title:
      print("In here")
      title = title.replace(', The', "", 1)
      title = "The " + title
    print(title)


    full_video_link = "https://www.youtube.com/watch?v="+youtube_id
    res = '240p'
    output_path = '../../ml_trailers'
    filename = ''

    counter = 0 
    #Find the imdb id for the movie trailer so that downloaded value 
    if title in movie_imdb_id_mapping:
      filename = str(movie_imdb_id_mapping[title])
    else:
      counter += 1
      print(counter)
      print("Can't find file count: ", str(counter))
      continue

    full_file = output_path+'/'+filename+'.mp4'
    
    
    if not os.path.exists(full_file):
      print("Downloading: " + filename + " to: " + output_path)


      try:
        correct_quality = YouTube(full_video_link).streams.filter(res=res, only_video=True).first()
      except:
        fail_download.append(title)
        print("Video unavailable")
        continue
      
      if correct_quality:
        try:
          correct_quality.download(output_path=output_path,filename=filename)
        except:
          print("Unable to download this video: " + full_video_link)

    else:
      print("This file exists: " + full_file)

  return fail_download

def get_trailer_imdb_id(movie_lens_dataset):
  mapping = {}

  for movieId, title in zip(movie_lens_dataset['imdbId'], movie_lens_dataset['Title']):
    mapping[title] = movieId

  return mapping

if __name__ == '__main__':
  movie_lens_dataset = pd.read_csv('MovieGenre.csv', encoding="ISO-8859-1")
  movie_imdb_id_mapping = get_trailer_imdb_id(movie_lens_dataset)
  

  #Asssuming this is run in data folder
  video_name_file = 'ml-youtube.csv'
  video_name_file = 'test.csv'

  video_names = pd.read_csv(video_name_file)

  make_directories(video_names)
  fail_download = download_videos(video_names, movie_imdb_id_mapping)

  with open('fail_download_log.txt', 'w') as f:
    for item in fail_download:
        f.write("%s\n" % item)



