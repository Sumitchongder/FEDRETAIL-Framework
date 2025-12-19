import numpy as np
import tensorflow as tf
from typing import Callable, List, Tuple


def horizontal_federated_training(
    global_model: tf.keras.Model,
    clients_X: List[np.ndarray],
    clients_y: List[np.ndarray],
    X_test,
    y_test,
    aggregation_fn: Callable,
    rounds: int = 200,
    local_epochs: int = 2,
    participation_prob: float = 1.0,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Horizontal Federated Learning (HFL) training loop.

    Parameters
    ----------
    aggregation_fn : Callable
        fedavg / fsvrg / coop
        Must return (weights, biases)

    Returns
    -------
    accuracy : list
        Global accuracy per round
    loss : list
        Global loss per round
    """
    accuracy, losses = [], []
    num_clients = len(clients_X)

    for rnd in range(rounds):
        local_weights, local_biases = [], []

        for i in range(num_clients):
            if np.random.rand() > participation_prob:
                continue

            local_model = tf.keras.models.clone_model(global_model)
            local_model.set_weights(global_model.get_weights())
            local_model.compile(
                optimizer=global_model.optimizer,
                loss=global_model.loss,
                metrics=global_model.metrics
            )

            local_model.fit(
                clients_X[i],
                clients_y[i],
                epochs=local_epochs,
                verbose=0
            )

            w, b = local_model.layers[1].get_weights()
            local_weights.append(w)
            local_biases.append(b)

        if len(local_weights) == 0:
            continue

        global_w, global_b = aggregation_fn(local_weights, local_biases)
        global_model.layers[1].set_weights([global_w, global_b])

        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
        accuracy.append(acc)
        losses.append(loss)

        if verbose:
            print(f"Round {rnd + 1}/{rounds} — Loss: {loss:.4f}, Acc: {acc:.4f}")

    return accuracy, losses
