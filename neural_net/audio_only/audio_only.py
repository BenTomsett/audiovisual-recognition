from pathlib import Path

import joblib
import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers.legacy import Adam
from keras.src.engine.input_layer import InputLayer
from keras.src.layers import Conv1D, MaxPooling1D, Activation
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import seaborn as sns


def load_mfcc_from_file(path):
    data = np.load(path)
    label = path.stem.split("_")[0]

    return data, label


def load_data(mfcc_dir):
    files = sorted(mfcc_dir.glob("*.npy"))
    n_files = len(list(files))

    if n_files == 0:
        print("No *.npy files found in", mfcc_dir)
        exit(1)

    loaded_data = [load_mfcc_from_file(file) for file in tqdm(files, total=n_files)]

    data, labels = zip(*loaded_data)

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
        epochs=100,
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
    mfcc_training_dir = Path("../../audio_features/mfccs/training/")
    mfcc_testing_dir = Path("../../audio_features/mfccs/testing/")

    training_data, training_labels = load_data(mfcc_training_dir)
    testing_data, testing_labels = load_data(mfcc_testing_dir)

    print("Training data samples:", len(training_data))
    print("Training data classes:", len(np.unique(training_labels)))
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
