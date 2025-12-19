import numpy as np
from typing import List, Tuple

def coop(
    local_weights: List[np.ndarray],
    local_biases: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CO-OP (Collaborative Learning) aggregation.

    Parameters
    ----------
    local_weights : List[np.ndarray]
        Client model weights
    local_biases : List[np.ndarray]
        Client model biases

    Returns
    -------
    global_weights : np.ndarray
    global_biases : np.ndarray
    """
    if len(local_weights) == 0:
        raise ValueError("local_weights list is empty")

    global_weights = np.mean(local_weights, axis=0)
    global_biases = np.mean(local_biases, axis=0)

    return global_weights, global_biases
