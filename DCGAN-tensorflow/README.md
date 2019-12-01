
* Compatible with TensorFlow 1.5.0
* Added parameters `grid_height` and `grid_width`: the size of the grid of the 'train' and 'test' images (in the folder 'samples')
* Parameters `input_height`, `input_width`, `output_height`, `output_width` are set automatically 
(assuming all images in the data set have the same size)
* Added `sample_rate` parameter: how often it creates a sample image ('1' for every iteration, '2' for every other iteration, etc.)

# DCGAN in Tensorflow

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*




## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)



## Usage

1. First you need to get the posters data, which can be downloaded from the dnn-movie-posters folder

Get posters data

Use flag -download to download the posters from Amazon (based on the URLs provided in MovieGenre.csv)

Use flag -resize to create smaller posters (30%, 40%, etc)

Use parameter -min_year=1980 to filter out the oldest movies.

`python3 get_data.py -download -resize`

2. Prepare dataset with the parameters you want:
`python3 prepare_dcgan_dataset.py -min_year=1980 -exclude_genres=Animation,Comedy,Family -ratio=60`

This will create a folder 'dcgan_movies_posters' with all the posters selected from the parameters values.


3. Move folder 'dcgan_movies_posters' to DCGAN-tensorflow/data/dcgan_movies_posters
4. In DCGAN-tensorflow, run the command with the parameters you need:

`python3 main.py --dataset dcgan_movies_posters --grid_height=6 --grid_width=10  -sample_rate=2 --train`

Other flags for more paramters can be seen in the main.py code.

## Related works

- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [DiscoGAN-pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
- [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
