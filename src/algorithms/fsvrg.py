import numpy as np
from typing import List, Tuple

def fsvrg(
    local_weights: List[np.ndarray],
    local_biases: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Federated SVRG-style aggregation.
    (As implemented in the FEDRETAIL reference code)

    Parameters
    ----------
    local_weights : List[np.ndarray]
        Client Dense-layer weights
    local_biases : List[np.ndarray]
        Client Dense-layer biases

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
