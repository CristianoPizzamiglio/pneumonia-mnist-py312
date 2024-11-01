from __future__ import annotations

from dataclasses import dataclass, field

import keras_tuner as kt
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.optimizers import RMSprop, Adam
from tensorflow.keras import layers, Model, Sequential

from utils import set_seed_

set_seed_()

mode_to_label = {0: "simple_model", 1: "pre_trained_model"}


@dataclass
class SimpleModelHyperParametersConfig:
    """
    Keras tuner hyperparameters.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters

    """

    hyper_parameters: kt.HyperParameters
    learning_rate: kt.HyperParameters.Choice = field(init=False)
    convolutional_layer_count: kt.HyperParameters.Choice = field(init=False)
    convolutional_first_layer_filter_count: kt.HyperParameters.Choice = field(
        init=False
    )
    dropout: kt.HyperParameters.Choice = field(init=False)
    dense_layer_neuron_count: kt.HyperParameters.Choice = field(init=False)

    def __post_init__(self) -> None:
        self.learning_rate = self.hyper_parameters.Choice(
            "learning_rate", values=[1e-4, 1e-3, 1e-2]
        )
        self.convolutional_layer_count = self.hyper_parameters.Choice(
            "convolutional_layer_count", [1, 2, 3]
        )
        self.convolutional_first_layer_filter_count = self.hyper_parameters.Choice(
            "convolutional_first_layer_filter_count", [16, 32, 48, 64]
        )
        self.dropout = self.hyper_parameters.Choice(
            "dropout", values=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        self.dense_layer_neuron_count = self.hyper_parameters.Choice(
            "dense_layer_neuron_count", values=[256, 512, 1024]
        )


@dataclass
class PreTrainedModelHyperParametersConfig:
    """
    Keras tuner hyperparameters.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters

    """

    hyper_parameters: kt.HyperParameters
    learning_rate: kt.HyperParameters.Choice = field(init=False)
    dropout: kt.HyperParameters.Choice = field(init=False)
    neuron_count: kt.HyperParameters.Choice = field(init=False)

    def __post_init__(self) -> None:
        self.learning_rate = self.hyper_parameters.Choice(
            "learning_rate", values=[1e-4, 1e-3, 1e-2]
        )
        self.dropout = self.hyper_parameters.Choice(
            "dropout", values=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        self.neuron_count = self.hyper_parameters.Choice(
            "neuron_count", values=[256, 512, 1024]
        )


def build_simple_model(hyper_parameters: kt.HyperParameters) -> Sequential:
    """
    Build a simple parametric convolutional neural network.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters

    Returns
    -------
    tf.keras.models.Sequential

    """
    hyper_parameters_specs = SimpleModelHyperParametersConfig(hyper_parameters)
    model = Sequential()

    for i in range(hyper_parameters_specs.convolutional_layer_count):
        filter_count = hyper_parameters_specs.convolutional_first_layer_filter_count
        if i == 0:
            model.add(
                layers.Conv2D(
                    filter_count, (3, 3), activation="relu", input_shape=(28, 28, 1)
                )
            )
        else:
            model.add(layers.Conv2D(filter_count * i * 2, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hyper_parameters_specs.dense_layer_neuron_count, activation="relu"
        )
    )
    model.add(layers.Dropout(hyper_parameters_specs.dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    adam_optimizer = Adam(learning_rate=hyper_parameters_specs.learning_rate)
    model.compile(
        optimizer=adam_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def build_pre_trained_model(
    hyper_parameters: kt.HyperParameters, image_size: int
) -> Model:
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
    hyper_parameters_ = PreTrainedModelHyperParametersConfig(hyper_parameters)

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
