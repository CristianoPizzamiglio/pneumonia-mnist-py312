from __future__ import annotations

import datetime
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import keras_tuner as kt
import tensorflow as tf
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.optimizers import RMSprop
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras import layers

from preprocessor import compute_datasets
from utils import set_seed_, to_dict

logdir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model.keras", monitor="val_loss", save_best_only=True
)

set_seed_()


@dataclass
class HyperParametersConfig:
    """
    Keras tuner hyperparameters.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters

    """

    hyper_parameters: kt.HyperParameters
    learning_rate: kt.HyperParameters.Choice = field(init=False)
    dropout: kt.HyperParameters.Float = field(init=False)
    neuron_count: kt.HyperParameters.Int = field(init=False)

    def __post_init__(self) -> None:
        self.learning_rate = self.hyper_parameters.Choice(
            "learning_rate", values=[1e-4, 1e-3, 1e-2]
        )
        self.dropout = self.hyper_parameters.Choice(
            "dropout", values=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        self.neuron_count = self.hyper_parameters.Choice(
            "neuron_count", values=[128, 256, 512, 1024]
        )


def build_model(hyper_parameters: kt.HyperParameters, image_size: int) -> Model:
    """
    Build a model starting from a pre-trained one. A parametric classifier is added.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters
    image_size : int

    Returns
    -------
    Model

    """
    hyper_parameters_ = HyperParametersConfig(hyper_parameters)

    pre_trained_model = InceptionV3(
        input_shape=(image_size, image_size, 3), include_top=False
    )
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer("mixed7")
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(hyper_parameters_.neuron_count, activation="relu")(x)
    x = layers.Dropout(hyper_parameters_.dropout)(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = Model(pre_trained_model.input, x)
    model.compile(
        optimizer=RMSprop(learning_rate=hyper_parameters_.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_model(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    epoch_count: int,
    image_size: int,
) -> None:
    """
    Tune the model.

    Parameters
    ----------
    training_dataset : Dataset
    validation_dataset : Dataset
    epoch_count : int
    image_size : int

    """
    tuner = kt.Hyperband(
        lambda hyper_parameters: build_model(hyper_parameters, image_size),
        objective="val_accuracy",
        max_epochs=epoch_count,
        project_name="pneumoniamnist",
    )
    tuner.search_space_summary(extended=True)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    callbacks = [tensorboard_callback, model_checkpoint_callback, stop_early]

    tuner.search(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
    )
    tuner.results_summary()

    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    with open(r"..\hyperparameter_tuning\best_hyperparameters.pkl", "wb") as file:
        pickle.dump(best_hyperparameters, file)


if __name__ == "__main__":
    param_to_value = to_dict(Path(r"..\config\params.yaml"))
    params_ = SimpleNamespace(**param_to_value)
    training_dataset_, validation_dataset_, test_dataset = compute_datasets(
        params_.image_size, params_.batch_size
    )
    tune_model(
        training_dataset_, validation_dataset_, params_.epoch_count, params_.image_size
    )
