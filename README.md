# Poster-GAN

The goal of poster-GAN is to train a generator that can be conditioned on a movie trailer to produce an 
appropriate poser.

## Dataset download

Run the following:

```
cd data
python3 download_dataset.py
```

## Running the training script

```
python tfgan_quick_train_main.py --run_name=<run_name>
```

Set the --storage_dir and --image_dir when not on cloud. 

