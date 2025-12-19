import tensorflow as tf
import numpy as np
from typing import List, Tuple


def vertical_federated_training(
    global_model: tf.keras.Model,
    clients_X: List[np.ndarray],
    clients_y: List[np.ndarray],
    X_test,
    y_test,
    rounds: int = 200,
    local_epochs: int = 2,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Vertical Federated Learning (VFL) training loop
    using gradient aggregation.
    """
    accuracy, losses = [], []

    for rnd in range(rounds):
        local_gradients = []

        for X, y in zip(clients_X, clients_y):
            local_model = tf.keras.models.clone_model(global_model)
            local_model.set_weights(global_model.get_weights())
            local_model.compile(
                optimizer=global_model.optimizer,
                loss=global_model.loss,
                metrics=global_model.metrics
            )

            local_model.fit(X, y, epochs=local_epochs, verbose=0)

            with tf.GradientTape() as tape:
                preds = local_model(X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)

            grads = tape.gradient(loss, local_model.trainable_variables)
            local_gradients.append(grads)

        aggregated_grads = [
            tf.reduce_mean(g, axis=0) for g in zip(*local_gradients)
        ]

        global_model.optimizer.apply_gradients(
            zip(aggregated_grads, global_model.trainable_variables)
        )

        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
        accuracy.append(acc)
        losses.append(loss)

        if verbose:
            print(f"Round {rnd + 1}/{rounds} — Loss: {loss:.4f}, Acc: {acc:.4f}")

    return accuracy, losses
