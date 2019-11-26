import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import numpy as np
import os

# This method is not serialized into the graph definition. Ergo, if this
# model is loaded where this module is not imported, it will error.
# We have to get off the .npy file format and use jpg in that case.
def tf_np_load(filepath):
        return tf_v1.convert_to_tensor(np.load(filepath))

def make_summary_example(movieId, poster_dir, trailer_dir):
    # Getting poster
    image_string = tf_v1.read_file(poster_dir + os.sep + movieId + '.jpg')
    image_decoded = tf_v1.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf_v1.image.resize_image_with_crop_or_pad(image_decoded, 256, 256)
    # image_decoded = tf_v2.expand_dims(image_decoded, 0)
    print("Poster shape debug:")
    print(image_decoded.shape)
    image = tf_v1.cast(image_decoded, tf_v1.float32)
    print(image.shape)
    image = tf_v1.image.resize(image, [64, 64], name="poster_resize")
    # 64 x 64 image with 3 channels
    print(image.shape)
    poster = tf_v1.reshape(image, [64, 64, 3], name="poster_reshape")

    # Getting trailer
    trailer_path = trailer_dir + os.sep + movieId + '.npy'
    # trailer_mat = tf_v1.py_func(tf_np_load, [trailer_path], tf_v1.float32)
    trailer_mat = tf_v1.zeros((240, 240, 3, 20))
    print(trailer_mat.shape)
    trailer_mat = tf_v1.reshape(trailer_mat, (240, 240, 3, 20))
    # trailer_img_paths = [os.path.join(trailer_path, f) for f in os.listdir(trailer_path) if
    #  os.path.isfile(os.path.join(trailer_path, f))]
    trailer_frames = []
    print(trailer_mat.shape)
    for i in range(tf_v1.shape(trailer_mat)[-1]):
        # image_string = tf_v1.read_file(trailer_img_path)
        # image_decoded = tf_v1.image.decode_jpeg(image_string, channels=3)
        # image_decoded = tf_v1.image.resize_image_with_crop_or_pad(image_decoded, 256, 256)
        # image_decoded = tf_v2.expand_dims(image_decoded, 0)
        image = trailer_mat[:, :, :, i]
        image = tf_v1.reshape(image, (240, 240, 3))
        image = tf_v1.cast(image, tf_v1.float32)
        # resized = tf_v1.image.resize(image, [64, 64])
        # trailer_frames.append(tf_v1.reshape(resized, [64, 64, 3]))
        trailer_frames.append(poster)

    return trailer_frames, poster

class SummaryDataset:
    def __init__(self, trailer_dir, poster_dir, folds=0):
        self.poster_dir = poster_dir
        self.trailer_dir = trailer_dir
        trailer_ids = set([str(f[:-4]) for f in os.listdir(trailer_dir) if
                       os.path.isfile(os.path.join(trailer_dir, f))])
        print("Number of Trailer IDs:", len(trailer_ids))
        poster_ids = set([str(f[:-4]) for f in os.listdir(poster_dir) if
             os.path.isfile(os.path.join(poster_dir, f))]) # remove the extension when comparing
        print("Number of Poster IDs:", len(poster_ids))
        self._movie_ids = list(trailer_ids.intersection(poster_ids))  # We only want examples where the data is complete
        print("IDs in training set: ", self._movie_ids)
        self._folds = folds
        if folds > 1:
            self._interval = len(self._movie_ids)/folds
            self._split_index = 0
        #self._data = [os.path.join(trailer_dir, movieId) for movieId in movieIds]
        #self._labels = [os.path.join(poster_dir, movieId) for movieId in movieIds]

    def make_example(self, movieId):
        return make_summary_example(movieId, self.poster_dir, self.trailer_dir)

    def get_split(self):
        if self._folds < 2:
            dataset = tf_v1.data.Dataset.from_tensor_slices(self._movie_ids)
            return dataset.map(self.make_example).cache().repeat(), None
        else:
            validation_ids = self._movie_ids[self._split_index*self._interval:(self._split_index + 1)*self._interval]
            train_ids = self._movie_ids[:self._split_index*self._interval] \
                        + self._movie_ids[(self._split_index + 1)*self._interval:]
            self._split_index += 1
            if self._split_index == self._folds:
                self._split_index = 0 # We've reached the end. Start over.
            train_dataset = tf_v1.data.Dataset.from_tensor_slices(train_ids).map(self.make_example)
            validation_dataset = tf_v1.data.Dataset.from_tensor_slices(validation_ids).map(self.make_example)
            return train_dataset, validation_dataset

