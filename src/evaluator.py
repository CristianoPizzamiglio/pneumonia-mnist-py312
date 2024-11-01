from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf

from builder import mode_to_label
from preprocessor import compute_datasets
from utils import set_seed_, to_dict

set_seed_()


def main(params: SimpleNamespace) -> None:
    """
    Evaluate model on the test set.

    Parameters
    ----------
    params : SimpleNamespace

    """

    _, _, test_dataset = compute_datasets(params)

    label = mode_to_label[params.mode]
    model = tf.keras.models.load_model(rf"..\training\{label}.keras")
    metric_to_value = model.evaluate(test_dataset, return_dict=True)
    print(metric_to_value)


if __name__ == "__main__":
    param_to_value = to_dict(Path(r"..\config\params.yaml"))
    params_ = SimpleNamespace(**param_to_value)
    main(params_)
