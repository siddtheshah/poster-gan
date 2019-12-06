# Download data set 
First you need to get the posters data, which can be downloaded from the dnn-movie-posters folder
Use flag -download to download the posters from Amazon (based on the URLs provided in MovieGenre.csv)

Use flag -resize to create smaller posters (30%, 40%, etc)

Use parameter -min_year=1980 to filter out the oldest movies.

`python3 get_data.py -download -resize`


# Prepare the data (formatting) for the GAN
Prepare dataset with the parameters you want: 
`python3 prepare_dcgan_dataset.py -min_year=1980 -exclude_genres=Animation,Comedy,Family -ratio=60`
