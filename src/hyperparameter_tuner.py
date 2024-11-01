from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import keras_tuner as kt
import tensorflow as tf
from tensorflow.data import Dataset

from builder import build_pre_trained_model, build_simple_model, mode_to_label
from preprocessor import compute_datasets
from utils import set_seed_, to_dict, get_callbacks

set_seed_()


def tune_model(
    training_dataset: Dataset, validation_dataset: Dataset, params: SimpleNamespace
) -> None:
    """
    Tune the model.

    Parameters
    ----------
    training_dataset : Dataset
    validation_dataset : Dataset
    params : SimpleNamespace

    """
    mode_to_model_builder_type = {
        0: build_simple_model,
        1: lambda hyper_parameters: build_pre_trained_model(
            hyper_parameters, params.image_size
        ),
    }
    label = mode_to_label[params.mode]

    tuner = kt.Hyperband(
        mode_to_model_builder_type[params.mode],
        objective="val_accuracy",
        max_epochs=params.epoch_count,
        project_name=label,
        directory=r"..\hyperparameter_tuning",
    )
    tuner.search_space_summary(extended=True)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)
    callbacks = get_callbacks(label)
    callbacks.append(stop_early)

    tuner.search(
        training_dataset,
        validation_data=validation_dataset,
        epochs=params.epoch_count,
        callbacks=callbacks,
    )
    tuner.results_summary()

    best_hyperparameters = tuner.get_best_hyperparameters()[0]

    with open(
        rf"..\hyperparameter_tuning\{label}_best_hyperparameters.pkl", "wb"
    ) as file:
        pickle.dump(best_hyperparameters, file)


if __name__ == "__main__":
    param_to_value = to_dict(Path(r"..\config\params.yaml"))
    params_ = SimpleNamespace(**param_to_value)
    training_dataset_, validation_dataset_, test_dataset = compute_datasets(params_)
    tune_model(training_dataset_, validation_dataset_, params_)
