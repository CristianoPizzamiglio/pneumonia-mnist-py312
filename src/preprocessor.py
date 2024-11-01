from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Tuple, Callable

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras import layers

from utils import set_seed_

set_seed_()


def compute_datasets(params: SimpleNamespace) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Preprocess data.

    Parameters
    ----------
    params : SimpleNamespace

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]

    """
    training_dataset, validation_dataset, test_dataset = tfds.load(
        "pneumonia_mnist", split=["train", "val", "test"], as_supervised=True
    )

    if params.mode == 0:
        preprocess_image_ = preprocess_image_simple_model
    else:
        image_size = [params.image_size, params.image_size]
        preprocess_image_ = functools.partial(
            preprocess_image_pre_trained_model, target_size=image_size
        )

    training_dataset = preprocess_dataset(
        training_dataset, preprocess_image_, params.batch_size
    )
    training_dataset = augment_data(training_dataset)
    validation_dataset = preprocess_dataset(
        validation_dataset, preprocess_image_, params.batch_size
    )
    test_dataset = preprocess_dataset(
        test_dataset, preprocess_image_, params.batch_size
    )

    return training_dataset, validation_dataset, test_dataset


def preprocess_dataset(
    dataset: Dataset, preprocess_image_: Callable, batch_size: int
) -> Dataset:
    """
    Preprocess dataset.

    Parameters
    ----------
    dataset : Dataset
    preprocess_image_ : Callable
    batch_size : int

    Returns
    -------
    Dataset

    """
    dataset = dataset.map(
        preprocess_image_, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def preprocess_image_pre_trained_model(
    image: tf.Tensor, label: tf.Tensor, target_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocess image for the pre-trained model.

    Parameters
    ----------
    image : tf.Tensor
    label : tf.Tensor
    target_size : Tuple[int, int]

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]

    """
    image /= 255
    image = tf.image.resize(image, target_size)
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def preprocess_image_simple_model(
    image: tf.Tensor, label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocess image.

    Parameters
    ----------
    image : tf.Tensor
    label : tf.Tensor

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]

    """
    image /= 255
    return image, label


def compute_augmentation_layers() -> Model:
    """
    Compute augmentation layers.

    Returns
    -------
    Model

    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation((-0.1, 0.1)),
            layers.RandomZoom((-0.1, 0.1)),
        ]
    )


def augment_data(dataset: Dataset) -> Dataset:
    """
    Augment data.

    Parameters
    ----------
    dataset : Dataset

    Returns
    -------
    Dataset

    """
    augmentation = compute_augmentation_layers()
    return dataset.map(lambda x_, y: (augmentation(x_, training=True), y))
