import cv2
import os
import numpy as np

def extract_videos_for_conv2d(video_input_file_path, feature_output_file_path, max_frames):
    if feature_output_file_path is not None:
        if os.path.exists(feature_output_file_path):
            return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success and count < max_frames:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 3000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA)
            channels = image.shape[2]
            for channel in range(channels):
                features.append(image[:, :, channel])
            count = count + 1
    unscaled_features = np.array(features)
    #print(unscaled_features)
    unscaled_features = np.transpose(unscaled_features, axes=(1, 2, 0))
    unscaled_features = np.reshape(unscaled_features, (240,240,3,count))
    print(unscaled_features.shape)
    if feature_output_file_path is not None:
        np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def scan_and_extract_videos_for_conv2d(data_dir_path, data_set_name=None, max_frames=None):
    if data_set_name is None:
        data_set_name = 'ml_trailers'
    if max_frames is None:
        max_frames = 20

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + data_set_name + '_key_frame_features'

    print(output_feature_data_dir_path)

    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    
    print(os.listdir(input_data_dir_path))

    for f in os.listdir(input_data_dir_path):
        print(f)
        video_file_path = input_data_dir_path + os.path.sep + f
        output_feature_file_path = output_feature_data_dir_path + os.path.sep + f.split('.')[0] + '.npy'

        if not os.path.exists(output_feature_file_path):
            x = extract_videos_for_conv2d(video_file_path, output_feature_file_path, max_frames)
            print(len(x), len(x[0]))
            y = f
            y_samples.append(y)
            x_samples.append(x)

    return x_samples, y_samples


def main():
    print(cv2.__version__)
    data_dir_path = '../../.'
    X, Y = scan_and_extract_videos_for_conv2d(data_dir_path)
    print(X[0].shape)


if __name__ == '__main__':
    main()