# Poster-GAN

The goal of poster-GAN is to train a generator that can be conditioned on a movie trailer to produce an 
appropriate poser.

## Dataset download and GAN training

To download the data and set up the GAN, enter the DCGAN-tensorflow folder, and follow the README steps there.


## Running the Summarizer training script

```
# Configure summarizer_config.json with paths and parameters first
python summarizer_main --train --eval --run_name=<some run name>
```

## File Overview:
- tfgan_quick_train_lib.py contains the discriminator and generator architecture. This was derived from the GAN tutorial provided in class.
- tfgan_quick_train_main.py contains the running of the gan on a specific dataset of images. The path to the images is currently hard-coded in the files.
- ./data/ contains a csv containing paths to the movielens movie posters. The download_dataset.py sript downloads the posters to a hard coded folder in the script
