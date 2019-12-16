import pandas as pd
from pytube import YouTube
import sys
import os
import cv2
from PIL import Image
import numpy as np


video_input_file_path = './toy_story.mp4'

if not os.path.exists(video_input_file_path):
  full_video_link = "https://www.youtube.com/watch?v=K26_sDKnvMU"
  res = "480p"

  correct_quality = YouTube(full_video_link).streams.filter(res=res, only_video=True).first()
  correct_quality.download(output_path='./',filename='toy_story')




vidcap = cv2.VideoCapture(video_input_file_path)
success, image = vidcap.read()
images = []
success = True

count = 0
max_frames = 7

while success and count < max_frames:
  vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 10000 + 7000))  # added this line
  success, image = vidcap.read()
  
  if success:
    image = cv2.cvtColor(cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    image = np.array(image)

    if count == 0:
      images = image
    else:
      images = np.concatenate((images, image), axis = 1)
    
    count +=1
images= np.array(images)
print(images.shape)

#images = np.reshape(images, (240,240*max_frames,3))
#print(images.shape)

Image.fromarray(np.array(images)).save("im_all" + str(count) +'.jpg' )
    
    