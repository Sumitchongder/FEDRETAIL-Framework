import numpy as np
from typing import List, Tuple


def iid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    shuffle: bool = True,
    seed: int | None = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    IID data split for Federated Learning.

    Parameters
    ----------
    X : np.ndarray
        Input features (e.g., images)
    y : np.ndarray
        Labels
    num_clients : int
        Number of federated clients
    shuffle : bool
        Whether to shuffle data before splitting
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    clients_X : List[np.ndarray]
        Feature splits per client
    clients_y : List[np.ndarray]
        Label splits per client
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    clients_X = np.array_split(X_shuffled, num_clients)
    clients_y = np.array_split(y_shuffled, num_clients)

    return clients_X, clients_y
