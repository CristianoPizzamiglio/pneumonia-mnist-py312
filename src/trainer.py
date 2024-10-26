from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

from tensorflow.data import Dataset
from tensorflow.keras import Model

from hyperparameter_tuner import (
    tensorboard_callback,
    model_checkpoint_callback,
    build_model,
)
from preprocessor import compute_datasets
from utils import set_seed_, to_dict

set_seed_()


def main(params: SimpleNamespace) -> None:
    """
    Entry point.

    Parameters
    ----------
    params : SimpleNamespace

    """

    training_dataset, validation_dataset, test_dataset = compute_datasets(
        params.image_size, params.batch_size
    )

    with open(r"..\hyperparameter_tuning\best_hyperparameters.pkl", "rb") as file:
        best_hyperparameters = pickle.load(file)
    model = build_model(best_hyperparameters, params.image_size)
    model.summary()

    fit_model(training_dataset, validation_dataset, model, params.epoch_count)
    metric_to_value = model.evaluate(test_dataset, return_dict=True)
    print(metric_to_value)


def fit_model(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    model: Model,
    epoch_count: int,
) -> None:
    """
    Fit model.

    Parameters
    ----------
    training_dataset : Dataset
    validation_dataset : Dataset
    model : Model
    epoch_count : int

    """
    callbacks = [tensorboard_callback, model_checkpoint_callback]
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
    )
    best_epoch_index = history.history["val_loss"].index(
        min(history.history["val_loss"])
    )
    print(
        f"Epoch Index: {best_epoch_index}\n"
        f"Validation Loss: {history.history['val_loss'][best_epoch_index]}\n"
        f"Validation Accuracy: {history.history['val_accuracy'][best_epoch_index]}"
    )

    with open("history.pkl", "wb") as file:
        pickle.dump(history.history, file)


if __name__ == "__main__":
    param_to_value = to_dict(Path(r"..\config\params.yaml"))
    params_ = SimpleNamespace(**param_to_value)
    main(params_)
