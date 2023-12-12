from pathlib import Path
import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv3D, MaxPooling3D
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def load_data_from_mat_file(file_path):
    data = loadmat(file_path)['tensorInput']
    label = file_path.stem.split('_')[0]
    return data, label


def load_data(training_data_path):
    files = sorted(training_data_path.glob('*.mat'))

    loaded_data = [load_data_from_mat_file(file) for file in files]
    data, labels = zip(*loaded_data)

    return np.array(data), np.array(labels)


if __name__ == '__main__':
    training_dir = Path('../video_features/training/')
    testing_dir = Path('../video_features/testing/')

    training_data, training_labels = load_data(training_dir)
    testing_data, testing_labels = load_data(testing_dir)

    label_encoder = LabelEncoder()
    training_labels_encoded = label_encoder.fit_transform(training_labels)
    testing_labels_encoded = label_encoder.transform(testing_labels)

    training_labels_categorical = to_categorical(training_labels_encoded)
    testing_labels_categorical = to_categorical(testing_labels_encoded)

    print("Shape of training data:", training_data.shape)
    print("Shape of training labels:", training_labels_categorical.shape)

    training_data = np.expand_dims(training_data, axis=-1)
    testing_data = np.expand_dims(testing_data, axis=-1)

    training_data = training_data / 255.0
    testing_data = testing_data / 255.0

    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(30, 100, 178, 1)),
        MaxPooling3D((2, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(20, activation='softmax')  # Assuming 20 classes
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_data, training_labels_categorical, epochs=10, batch_size=32, validation_split=0.2)

    test_loss, test_accuracy = model.evaluate(testing_data, testing_labels_categorical)
    print(f"Test accuracy: {test_accuracy}")
