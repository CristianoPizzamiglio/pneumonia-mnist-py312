import random
from pathlib import Path
from typing import Any, Dict

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
