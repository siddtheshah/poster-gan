import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import summarizer.graph
import numpy as np
import os
import imghdr

global summary_graph
summary_graph = summarizer.graph.summary_graph

# This method is not serialized into the graph definition. Ergo, if this
# model is loaded where this module is not imported, it will error.
# We have to get off the .npy file format and use jpg in that case.
def tf_np_load(filepath):
    return tf_v1.convert_to_tensor(np.load(filepath).astype(np.float32))

def make_summary_example(movieId, poster_dir, trailer_dir):
    image_string = tf_v1.read_file(poster_dir + os.sep + movieId + '.jpg')
    image_decoded = tf_v1.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf_v1.image.convert_image_dtype(image_decoded, tf_v1.float32)
    image_decoded = tf_v1.image.resize_image_with_crop_or_pad(image_decoded, 256, 256)
    poster = tf_v1.image.resize(image_decoded, [64, 64], name="poster_resize")
    # 64 x 64 image with 3 channels
    # print(image.shape)
    # poster = tf_v1.reshape(image, [64, 64, 3], name="poster_reshape")

    # Getting trailer
    trailer_path = trailer_dir + os.sep + movieId + '.npy'
    trailer_mat = tf_np_load(trailer_path)
    trailer_mat = tf_v1.reshape(trailer_mat, (240, 240, 3, 20))
    # trailer_img_paths = [os.path.join(trailer_path, f) for f in os.listdir(trailer_path) if
    #  os.path.isfile(os.path.join(trailer_path, f))]
    trailer_frames = []
    for i in range(20):
        image = trailer_mat[:, :, :, i]
        image = tf_v1.reshape(image, (240, 240, 3))
        image = tf_v1.cast(image, tf_v1.float32)
        resized = tf_v1.image.resize(image, [64, 64])
        trailer_frames.append(tf_v1.reshape(resized, [64, 64, 3]))

    return trailer_frames, poster

def make_example(trailer_dir, poster_dir):
    def example(movieId):
        # with summary_graph.as_default() as graph:
        # Getting poster
        movieId = tf_v1.Print(movieId, [movieId])
        image_string = tf_v1.read_file(poster_dir + os.sep + movieId + '.jpg')
        image_decoded = tf_v1.image.decode_jpeg(image_string, channels=3)
        image_decoded = tf_v1.image.resize_image_with_crop_or_pad(image_decoded, 256, 256)
        # image_decoded = tf_v2.expand_dims(image_decoded, 0)
        # print("Poster shape debug:")
        # print(image_decoded.shape)
        image = tf_v1.cast(image_decoded, tf_v1.float32)
        # print(image.shape)
        image = tf_v1.image.resize(image, [64, 64], name="poster_resize")
        # 64 x 64 image with 3 channels
        # print(image.shape)
        poster = tf_v1.reshape(image, [64, 64, 3], name="poster_reshape")

        # Getting trailer
        trailer_path = trailer_dir + os.sep + movieId + '.npy'
        trailer_mat = tf_v1.py_func(tf_np_load, [trailer_path], tf_v1.float32)
        trailer_mat = tf_v1.reshape(trailer_mat, (240, 240, 3, 20))
        trailer_frames = []
        for i in range(20):
            image = trailer_mat[:, :, :, i]
            image = tf_v1.reshape(image, (240, 240, 3))
            image = tf_v1.cast(image, tf_v1.float32)
            resized = tf_v1.image.resize(image, [64, 64])
            trailer_frames.append(tf_v1.reshape(resized, [64, 64, 3]))

        return trailer_frames, poster
    return example

def valid_poster(path):
    return os.path.isfile(path)  and imghdr.what(path) == 'jpeg'

def get_useable_ids(trailer_dir, poster_dir):
    trailer_ids = set()
    for filename in os.listdir(trailer_dir):
        path = os.path.join(trailer_dir, filename)
        if os.path.isfile(path):
            arr = np.load(path)
            if arr.size == 20 * 240 * 240 * 3:
                trailer_ids.add(filename[:-4])

    print("Trailer IDS", trailer_ids)
    print("Number of Trailer IDs:", len(trailer_ids))
    # remove the extension when comparing
    movie_ids = []
    for filename in os.listdir(poster_dir):
        raw_id = str(filename[:-4])
        if raw_id in trailer_ids:
            if valid_poster(os.path.join(poster_dir, filename)):
                movie_ids.append(raw_id)

    print("IDs in training set: ", movie_ids)
    return movie_ids

def create_summary_dataset(trailer_dir, poster_dir, batch_size):
    movie_ids = get_useable_ids(trailer_dir, poster_dir)

    total = len(movie_ids)
    train_total = int(.7*total)
    train_dataset = tf_v1.data.Dataset.from_tensor_slices(movie_ids[:train_total]).shuffle(total).map(make_example(trailer_dir, poster_dir)).batch(batch_size).cache()
    validation_dataset = tf_v1.data.Dataset.from_tensor_slices(movie_ids[train_total:]).shuffle(total).map(
        make_example(trailer_dir, poster_dir)).batch(batch_size).cache()
    return train_dataset, validation_dataset

