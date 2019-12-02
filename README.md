# Poster-GAN

The goal of poster-GAN is to train a generator that can be conditioned on a movie trailer to produce an 
appropriate poster.

## Movie Poster Download and GAN training

To download the data and setup the GAN, enter the DCGAN-tensorflow folder, and follow the README steps there. The movie posters can also be independently downloaded via ./data/download_posters.py script.

## Movie Trailer Download and Frame Extraction
To download the corresponding movie trailers script, run ./data/download_ml-youtube.py. To extract frames from the movie trailers for the summarizer, run ./data/frame_extractor.py.

## Running the Summarizer training script

```
# Configure summarizer_config.json with paths and parameters first
python summarizer_main --train --eval --run_name=<some run name>
# If doing mock training, i.e. no generator/discriminator module, specify --mock.
```

## Directory Structure Overview:
./data contains scripts to download and extract the movie posters and videos for the project.

./DCGAN-Tensorflow contains the model structure and functions to run and uncoditional GAN to create movie posters.

./summarizer contains the library for the summarizer architecture

./generator contains prior, basic versions of unconditioned GANs. These were unsuccessful on the movie poster task and can be ignored. To run the GAN simply run the main file

