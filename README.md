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
Edit the tfgan_quick_train_lib.py file to set the RUN_NAME. Depending on where you want the model
info, change the paths for MODEL_DIR, RESULTS_DIR, etc.

```
python tfgan_quick_train_main.py
```

