import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

from src.models import create_neural_network
from src.data import iid_split
from src.algorithms import fedavg, fsvrg, coop
from src.training import horizontal_federated_training
from src.utils.visualization import plot_multiple_metrics


def run_algorithm_comparison():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    num_clients = 10
    rounds = 50

    clients_X, clients_y = iid_split(X_train, y_train, num_clients)

    algorithms = {
        "FedAvg": fedavg,
        "FSVRG": fsvrg,
        "CO-OP": coop
    }

    results = {}

    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        model = create_neural_network()

        acc, _ = horizontal_federated_training(
            global_model=model,
            clients_X=clients_X,
            clients_y=clients_y,
            X_test=X_test,
            y_test=y_test,
            aggregation_fn=algo,
            rounds=rounds,
            local_epochs=2
        )

        results[name] = acc

    plot_multiple_metrics(
        metrics=list(results.values()),
        labels=list(results.keys()),
        ylabel="Accuracy",
        title="FedAvg vs FSVRG vs CO-OP"
    )


if __name__ == "__main__":
    run_algorithm_comparison()
