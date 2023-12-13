import os
from pathlib import Path

import joblib
import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv3D, MaxPooling3D
from keras.optimizers.legacy import Adam
from keras.src.engine.input_layer import InputLayer
from keras.src.layers import Conv1D, Activation, MaxPooling1D
from keras.utils import to_categorical
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_mfcc_from_file(path):
    data = np.load(path)

    return data


def load_data_from_mat_file(file_path):
    data = loadmat(file_path)["tensorInput"]
    return data


def combine_audio_visual_features(mfcc, dct):
    scaler_mfcc = StandardScaler()
    mfcc_normalized = scaler_mfcc.fit_transform(mfcc.T).T

    scaler_dct = StandardScaler()
    dct_reshaped = dct.reshape(dct.shape[0], -1)
    dct_normalized = scaler_dct.fit_transform(dct_reshaped).reshape(dct.shape)

    dct_time_avg = np.mean(dct_normalized, axis=0)

    dct_time_avg_flattened = dct_time_avg.flatten()

    target_length = mfcc.shape[1] * mfcc.shape[0]
    if dct_time_avg_flattened.size > target_length:
        dct_time_avg_resized = dct_time_avg_flattened[:target_length]
    else:
        dct_time_avg_resized = np.pad(
            dct_time_avg_flattened,
            (0, target_length - dct_time_avg_flattened.size),
            "constant",
        )

    dct_time_avg_reshaped = dct_time_avg_resized.reshape(mfcc.shape)

    return np.concatenate((mfcc_normalized, dct_time_avg_reshaped), axis=1)


def load_data(video_feature_path, audio_feature_path):
    files = sorted(video_feature_path.glob("*.mat"))

    data = []
    labels = []

    for file in files:
        stem = file.stem
        label = stem.split("_")[0]

        audio_file = audio_feature_path / (stem + ".npy")

        if audio_file.exists():
            mfcc = load_mfcc_from_file(audio_file)
            dct = load_data_from_mat_file(file)

            data.append(combine_audio_visual_features(mfcc, dct))
            labels.append(label)

    return np.array(data), np.array(labels)


def build_layers(n_classes, input_shape):
    model = Sequential()

    model.add(InputLayer(input_shape=(input_shape[0], input_shape[1])))
    model.add(Conv1D(32, 3, activation="elu"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation("relu"))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    return model


def train_model(training_data, training_labels):
    train_data, val_data, train_labels, val_labels = train_test_split(
        training_data, training_labels, test_size=0.2
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)

    train_labels_enc = label_encoder.transform(train_labels)
    val_labels_enc = label_encoder.transform(val_labels)

    train_labels = to_categorical(train_labels_enc)
    val_labels = to_categorical(val_labels_enc)

    model = build_layers(len(np.unique(training_labels)), training_data[0].shape)

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer
    )

    model.summary()

    history = model.fit(
        train_data,
        train_labels,
        batch_size=16,
        epochs=40,
        validation_data=(val_data, val_labels),
    )

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Training Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    model.save_weights(Path("./model.h5"))
    joblib.dump(label_encoder, Path("./label_encoder.pkl"))

    return model, label_encoder


def test_model(model, testing_data, testing_labels, label_encoder):
    # Convert labels to categorical
    test_labels_enc = label_encoder.transform(testing_labels)
    test_labels_categorical = to_categorical(test_labels_enc)

    # Make predictions
    predictions = model.predict(testing_data)

    # Convert predictions to label indices
    predicted_labels_enc = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels_enc, predicted_labels_enc)
    print("Test Accuracy:", accuracy)

    # Confusion matrix
    cm = confusion_matrix(test_labels_enc, predicted_labels_enc)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def load_model(n_classes, input_shape, training_data, training_labels):
    path = Path("./model.h5")

    if path.exists():
        model = build_layers(n_classes, input_shape)
        model.compile(
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            optimizer=Adam(learning_rate=0.001),
        )
        model.load_weights(path)

        label_encoder = joblib.load(Path("./label_encoder.pkl"))

        return model, label_encoder
    else:
        return train_model(training_data, training_labels)


def main():
    audio_training_dir = Path("../../audio_features/mfccs/training/")
    video_training_dir = Path("../../video_features/training/")
    audio_testing_dir = Path("../../audio_features/mfccs/testing/")
    video_testing_dir = Path("../../video_features/testing/")

    training_data, training_labels = load_data(video_training_dir, audio_training_dir)
    testing_data, testing_labels = load_data(video_testing_dir, audio_testing_dir)

    print("Training data samples:", len(training_data))
    print("Training data classes:", len(np.unique(training_labels)))
    print("Training data shape:", training_data[0].shape)
    print("Testing data samples:", len(testing_data))
    print("Testing data classes:", len(np.unique(testing_labels)))

    model, label_encoder = load_model(
        len(np.unique(training_labels)),
        training_data[0].shape,
        training_data,
        training_labels,
    )

    test_model(model, testing_data, testing_labels, label_encoder)


if __name__ == "__main__":
    main()
