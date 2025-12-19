import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

from src.models import create_neural_network
from src.data import iid_split, non_iid_split
from src.algorithms import fedavg
from src.training import horizontal_federated_training
from src.utils.visualization import plot_multiple_metrics


def run_hfl_experiments():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    num_clients = 10
    rounds = 50

    iid_clients = iid_split(X_train, y_train, num_clients)
    non_iid_clients = non_iid_split(X_train, y_train, num_clients)

    scenarios = {
        "IID": iid_clients,
        "Non-IID": non_iid_clients
    }

    results = {}

    for name, (clients_X, clients_y) in scenarios.items():
        print(f"\nRunning HFL ({name})...")
        model = create_neural_network()

        acc, _ = horizontal_federated_training(
            global_model=model,
            clients_X=clients_X,
            clients_y=clients_y,
            X_test=X_test,
            y_test=y_test,
            aggregation_fn=fedavg,
            rounds=rounds,
            local_epochs=2,
            participation_prob=0.8
        )

        results[name] = acc

    plot_multiple_metrics(
        metrics=list(results.values()),
        labels=list(results.keys()),
        ylabel="Accuracy",
        title="FEDRETAIL: IID vs Non-IID"
    )


if __name__ == "__main__":
    run_hfl_experiments()
