# Poster-GAN

The goal of poster-GAN is to train a generator that can be conditioned on a movie trailer to produce an 
appropriate poser.

## Dataset download

Run the following:

```
cd data
python3 download_dataset.py
```

## Running the GAN training script

```
cd generator
python tfgan_quick_train_main.py --help      # to list all arguments
python tfgan_quick_train_main.py --run_name=<run_name>
```

## Running the Summarizer training script

```
# Configure summarizer_config.json with paths and parameters first
python summarizer_main --train --eval --run_name=<some run name>
```

## File Overview:
- tfgan_quick_train_lib.py contains the discriminator and generator architecture. This was derived from the GAN tutorial provided in class.
- tfgan_quick_train_main.py contains the running of the gan on a specific dataset of images. The path to the images is currently hard-coded in the files.
- ./data/ contains a csv containing paths to the movielens movie posters. The download_dataset.py sript downloads the posters to a hard coded folder in the script
