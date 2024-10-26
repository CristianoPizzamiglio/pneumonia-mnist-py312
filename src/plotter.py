from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_image(image: np.ndarray, one_hot_label: int) -> None:
    """
    Plot an image.

    Parameters
    ----------
    image : np.ndarray
    one_hot_label : int

    """
    plt.imshow(image, cmap="gray")
    label = "Healthy" if one_hot_label == 0 else "Pneumonia"
    plt.title(label)
    plt.axis("off")
    plt.show()
