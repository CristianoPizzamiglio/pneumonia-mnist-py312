from __future__ import annotations

import datetime
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import yaml


def set_seed_() -> None:
    """Set seed."""

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_dict(path: Path) -> Dict[Any, Any]:
    """
    Import yaml file and convert it to a dictionary.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Dict[Any, Any]

    Raises
    ------
    FileNotFoundError
    IOError

    """
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file.read())

    except FileNotFoundError:
        raise FileNotFoundError()

    except IOError:
        raise IOError()


# TODO Add `export_dir` param
def get_callbacks(label: str) -> List:
    """
    Get callbacks.

    Parameters
    ----------
    label : str

    Returns
    -------
    List

    """
    logdir = os.path.join(
        rf"..\hyperparameter_tuning\{label}_runs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=rf"..\hyperparameter_tuning\{label}.keras",
        monitor="val_accuracy",
        save_best_only=True,
    )
    return [tensorboard_callback, model_checkpoint_callback]
