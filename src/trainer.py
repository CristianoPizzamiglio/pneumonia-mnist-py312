from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

from tensorflow.data import Dataset
from tensorflow.keras import Model

from builder import build_pre_trained_model, mode_to_label, build_simple_model
from preprocessor import compute_datasets
from utils import set_seed_, to_dict, get_callbacks

set_seed_()


def main(params: SimpleNamespace) -> None:
    """
    Entry point.

    Parameters
    ----------
    params : SimpleNamespace

    """

    training_dataset, validation_dataset, test_dataset = compute_datasets(params)
    label = mode_to_label[params.mode]

    with open(
        rf"..\hyperparameter_tuning\{label}_best_hyperparameters.pkl", "rb"
    ) as file:
        best_hyperparameters = pickle.load(file)

    if params.mode == 0:
        model = build_simple_model(best_hyperparameters)
    else:
        model = build_pre_trained_model(best_hyperparameters, params.image_size)
    model.summary()

    fit_model(training_dataset, validation_dataset, model, params.epoch_count, label)


def fit_model(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    model: Model,
    epoch_count: int,
    label: str,
) -> None:
    """
    Fit model.

    Parameters
    ----------
    training_dataset : Dataset
    validation_dataset : Dataset
    model : Model
    epoch_count : int
    label : str

    """
    callbacks = get_callbacks(label)
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
    )
    best_epoch_index = history.history["val_accuracy"].index(
        max(history.history["val_accuracy"])
    )
    print(
        f"Epoch Index: {best_epoch_index}\n"
        f"Validation Loss: {history.history['val_loss'][best_epoch_index]}\n"
        f"Validation Accuracy: {history.history['val_accuracy'][best_epoch_index]}"
    )

    with open(rf"..\training\{label}_history.pkl", "wb") as file:
        pickle.dump(history.history, file)


if __name__ == "__main__":
    param_to_value = to_dict(Path(r"..\config\params.yaml"))
    params_ = SimpleNamespace(**param_to_value)
    main(params_)
