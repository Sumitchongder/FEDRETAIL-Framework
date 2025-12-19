#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Part 1 – Comparison of Federated Learning Algorithms required for FEDRETAIL Framework 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the number of clients and local epochs
num_clients = 10
local_epochs = 2

# Shuffle the data before splitting
shuffled_indices = np.random.permutation(len(train_images))
shuffled_train_images = train_images[shuffled_indices]
shuffled_train_labels = train_labels[shuffled_indices]

# Split the shuffled data into equal partitions for each client
split_train_images = np.array_split(shuffled_train_images, num_clients)
split_train_labels = np.array_split(shuffled_train_labels, num_clients)

# Initialize global models for federated learning algorithms
global_model_fedavg = create_model()
global_model_fsvrg = create_model()
global_model_coop = create_model()

# Initialize lists to store accuracy and loss for each algorithm
accuracy_fedavg = []
accuracy_fsvrg = []
accuracy_coop = []
losses_fedavg = []
losses_fsvrg = []
losses_coop = []

# Federated Learning with specified algorithms
for epoch in range(200):
    local_weights = []
    local_biases = []
    local_histories = []
    for i in range(num_clients):
        # Create and compile local model
        local_model = create_model()
        local_model.set_weights(global_model_fedavg.get_weights())  # Use FedAvg model weights for initialization

        # Train local model
        history = local_model.fit(split_train_images[i], split_train_labels[i], epochs=local_epochs, verbose=0)
        local_histories.append(history.history)

        # Get local model weights
        local_weights.append(local_model.layers[1].get_weights()[0])
        local_biases.append(local_model.layers[1].get_weights()[1])

    # Aggregate local weights using Federated Averaging (FedAvg)
    global_weights_fedavg = np.mean(local_weights, axis=0)
    global_biases_fedavg = np.mean(local_biases, axis=0)
    global_model_fedavg.layers[1].set_weights([global_weights_fedavg, global_biases_fedavg])

    # Aggregate local weights using Federated Stochastic Variance Reduced Gradient (FSVRG)
    global_weights_fsvrg = np.mean(local_weights, axis=0)
    global_biases_fsvrg = np.mean(local_biases, axis=0)
    global_model_fsvrg.layers[1].set_weights([global_weights_fsvrg, global_biases_fsvrg])

    # Aggregate local weights using CO-OP algorithm
    global_weights_coop = np.mean(local_weights, axis=0)
    global_biases_coop = np.mean(local_biases, axis=0)
    global_model_coop.layers[1].set_weights([global_weights_coop, global_biases_coop])

    # Evaluate global models
    loss_fedavg, accuracy_fedavg_epoch = global_model_fedavg.evaluate(test_images, test_labels, verbose=0)
    loss_fsvrg, accuracy_fsvrg_epoch = global_model_fsvrg.evaluate(test_images, test_labels, verbose=0)
    loss_coop, accuracy_coop_epoch = global_model_coop.evaluate(test_images, test_labels, verbose=0)

    # Store accuracy and loss for each epoch
    accuracy_fedavg.append(accuracy_fedavg_epoch)
    accuracy_fsvrg.append(accuracy_fsvrg_epoch)
    accuracy_coop.append(accuracy_coop_epoch)
    losses_fedavg.append(loss_fedavg)
    losses_fsvrg.append(loss_fsvrg)
    losses_coop.append(loss_coop)
    
    # Print epoch
    print(f"Epoch {epoch + 1}/{200}")

# Plot comparison of accuracy for the three algorithms
plt.figure(figsize=(10, 6))
plt.title('Accuracy Comparisons')
plt.plot(range(1, len(accuracy_fedavg) + 1), accuracy_fedavg, label='FedAvg')
plt.plot(range(1, len(accuracy_fsvrg) + 1), accuracy_fsvrg, label='FSVRG')
plt.plot(range(1, len(accuracy_coop) + 1), accuracy_coop, label='CO-OP')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot comparison of loss for the three algorithms
plt.figure(figsize=(10, 6))
plt.title('Loss Comparisons')
plt.plot(range(1, len(losses_fedavg) + 1), losses_fedavg, label='FedAvg')
plt.plot(range(1, len(losses_fsvrg) + 1), losses_fsvrg, label='FSVRG')
plt.plot(range(1, len(losses_coop) + 1), losses_coop, label='CO-OP')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:

#Part 2 – Implementation of FEDRETAIL FRAMEWORK with Horizontal Federated Learning & NN Model 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the number of clients and local epochs
num_clients = 10
local_epochs = 5

# Shuffle the data before splitting
shuffled_indices = np.random.permutation(len(train_images))
shuffled_train_images = train_images[shuffled_indices]
shuffled_train_labels = train_labels[shuffled_indices]

# Split the shuffled data into equal partitions for each client
split_train_images = np.array_split(shuffled_train_images, num_clients)
split_train_labels = np.array_split(shuffled_train_labels, num_clients)

# Print the class distribution for each client
for i, (images, labels) in enumerate(zip(split_train_images, split_train_labels)):
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print(f"Retailer {i+1} - Class Distribution: {dict(zip(unique_classes, class_counts))}")


# In[ ]:


# Mapping of class labels to class names for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Print the class distribution for Retailer 1 and plot it
retailer_index = 0  # Retailer 1 index is 0
unique_classes, class_counts = np.unique(split_train_labels[retailer_index], return_counts=True)

# Replace labels with class names
class_names_dict = {class_idx: class_name for class_idx, class_name in enumerate(class_names)}
class_names_labels = [class_names_dict[label] for label in unique_classes]

plt.figure(figsize=(10, 5))
plt.bar(class_names_labels, class_counts, color='skyblue')
plt.title(f"Class Distribution for Retailer {retailer_index+1} [IID Distribution]")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the number of clients and local epochs
num_clients = 10
local_epochs = 2

# Shuffle the data before splitting
shuffled_indices = np.random.permutation(len(train_images))
shuffled_train_images = train_images[shuffled_indices]
shuffled_train_labels = train_labels[shuffled_indices]

# Split the shuffled data into equal partitions for each client
split_train_images = np.array_split(shuffled_train_images, num_clients)
split_train_labels = np.array_split(shuffled_train_labels, num_clients)

# Initialize global model for FEDCOM
global_model_fedretail = create_model()

# Initialize lists to store accuracy for each scenario
accuracy = [[] for _ in range(6)]
losses = [[] for _ in range(6)]

# Centralized training
centralized_model = create_model()
history_centralized = centralized_model.fit(train_images, train_labels, epochs=200, 
                                            validation_data=(test_images, test_labels), verbose=1)
accuracy[4].append(history_centralized.history['val_accuracy'])
losses[4].append(history_centralized.history['loss'])

# FEDRETAIL Implementing (Horizontal FL)
for epoch in range(200):
    local_weights = []
    local_biases = []
    for i in range(num_clients):
        # Create and compile local model
        local_model = create_model()
        local_model.set_weights(global_model_fedretail.get_weights())

        # Train local model
        local_model.fit(split_train_images[i], split_train_labels[i], epochs=local_epochs, verbose=0)

        # Get local model weights
        local_weights.append(local_model.layers[1].get_weights()[0])
        local_biases.append(local_model.layers[1].get_weights()[1])

    # Aggregate local weights
    global_weights = np.mean(local_weights, axis=0)
    global_biases = np.mean(local_biases, axis=0)

    # Update global model with aggregated weights
    global_model_fedretail.layers[1].set_weights([global_weights, global_biases])

    # Evaluate global model
    loss, accuracy_fedretail = global_model_fedretail.evaluate(test_images, test_labels, verbose=0)
    accuracy[5].append(accuracy_fedretail)
    losses[5].append(loss)
    print(f"FEDRETAIL - Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy_fedretail}")

# Training for specific retailers
for retailer_index in [0, 2, 6, 8]:  # Retailer 1, 3, 7, 9
    local_model = create_model()
    history = local_model.fit(split_train_images[retailer_index], split_train_labels[retailer_index], 
                              epochs=200, validation_data=(test_images, test_labels), verbose=1)
    accuracy[retailer_index//2].append(history.history['val_accuracy'])
    losses[retailer_index//2].append(history.history['loss'])
    
labels = ['Retailer 1', 'Retailer 3', 'Retailer 7', 'Retailer 9', 'Centralized FL', 'FEDRETAIL']

# Plot comparison of accuracy
plt.figure(figsize=(10, 6))
plt.title('Accuracy Comparisons')
for i in range(len(accuracy)):
    if i == 2:
        if accuracy[i] and isinstance(accuracy[i][0], (list, np.ndarray)):
            plt.plot(range(1, len(accuracy[i][0]) + 1), accuracy[i][0], label=labels[i])
    elif i == 5:
        if accuracy[i] and isinstance(accuracy[i], (list, np.ndarray)):  # Corrected this line
            plt.plot(range(1, len(accuracy[i]) + 1), accuracy[i], label=labels[i])
    elif accuracy[i] and isinstance(accuracy[i][0], (list, np.ndarray)):
        plt.plot(range(1, len(accuracy[i][0]) + 1), accuracy[i][0], label=labels[i])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot comparison of loss
plt.figure(figsize=(10, 6))
plt.title('Loss Comparisons')
for i in range(len(losses)):
    if i == 2:
        if losses[i] and isinstance(losses[i][0], (list, np.ndarray)):
            plt.plot(range(1, len(losses[i][0]) + 1), losses[i][0], label=labels[i])
    elif i == 5:
        if losses[i] and isinstance(losses[i], (list, np.ndarray)):  # Corrected this line
            plt.plot(range(1, len(losses[i]) + 1), losses[i], label=labels[i])
    elif losses[i] and isinstance(losses[i][0], (list, np.ndarray)):
        plt.plot(range(1, len(losses[i][0]) + 1), losses[i][0], label=labels[i])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:

# Part 3 – Implementation of FEDRETAIL with Vertical Federated Learning using Logistic Regression 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers, regularizers
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(10, activation='softmax')
    ])
    optimizer = optimizers.Adam(learning_rate=0.001)  # Adjusted learning rate
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the number of clients and local epochs
num_clients = 10
local_epochs = 2

# Shuffle the data before splitting
shuffled_indices = np.random.permutation(len(train_images))
shuffled_train_images = train_images[shuffled_indices]
shuffled_train_labels = train_labels[shuffled_indices]

# Split the shuffled data into equal partitions for each client
split_train_images = np.array_split(shuffled_train_images, num_clients)
split_train_labels = np.array_split(shuffled_train_labels, num_clients)

# Initialize global model for FEDCOM
global_model_fedretail = create_model()

# Initialize lists to store accuracy for each scenario
accuracy = [[] for _ in range(6)]
losses = [[] for _ in range(6)]

# Centralized training
centralized_model = create_model()
history_centralized = centralized_model.fit(train_images, train_labels, epochs=200,
                                            validation_data=(test_images, test_labels), verbose=1)
accuracy[4].append(history_centralized.history['val_accuracy'])
losses[4].append(history_centralized.history['loss'])

# FEDRETAIL Implementing (Vertical FL)
for epoch in range(200):
    local_gradients = []
    for i in range(num_clients):
        # Create and compile local model
        local_model = create_model()
        local_model.set_weights(global_model_fedretail.get_weights())

        # Train local model
        local_model.fit(split_train_images[i], split_train_labels[i], epochs=local_epochs, verbose=0)

        # Compute gradients of local model
        with tf.GradientTape() as tape:
            predictions = local_model(split_train_images[i])
            loss = tf.keras.losses.sparse_categorical_crossentropy(split_train_labels[i], predictions)
        gradients = tape.gradient(loss, local_model.trainable_variables)

        # Clip gradients if necessary
        # gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

        local_gradients.append(gradients)

    # Aggregate local gradients
    aggregated_gradients = []
    for grads_list in zip(*local_gradients):
        aggregated_grads = tf.reduce_mean(grads_list, axis=0)
        aggregated_gradients.append(aggregated_grads)

    # Apply aggregated gradients to global model
    global_model_fedretail.optimizer.apply_gradients(zip(aggregated_gradients, 
                                                         global_model_fedretail.trainable_variables))

    # Evaluate global model
    loss, accuracy_fedretail = global_model_fedretail.evaluate(test_images, test_labels, verbose=0)
    accuracy[5].append(accuracy_fedretail)
    losses[5].append(loss)
    print(f"FEDRETAIL - Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy_fedretail}")

# Training for specific retailers
for retailer_index in [0, 2, 6, 8]:  # Retailer 1, 3, 7, 9
    local_model = create_model()
    history = local_model.fit(split_train_images[retailer_index], split_train_labels[retailer_index], 
                              epochs=200, validation_data=(test_images, test_labels), verbose=1)
    accuracy[retailer_index//2].append(history.history['val_accuracy'])
    losses[retailer_index//2].append(history.history['loss'])

labels = ['Retailer 1', 'Retailer 3', 'Retailer 7', 'Retailer 9', 'FEDRETAIL', 'Centralized FL']

# Plot comparison of accuracy
plt.figure(figsize=(10, 6))
plt.title('Accuracy Comparisons')
for i in range(len(accuracy)):
    if i == 2:
        if accuracy[i] and isinstance(accuracy[i][0], (list, np.ndarray)):
            plt.plot(range(1, len(accuracy[i][0]) + 1), accuracy[i][0], label=labels[i])
    elif i == 5:
        if accuracy[i] and isinstance(accuracy[i], (list, np.ndarray)):  # Corrected this line
            plt.plot(range(1, len(accuracy[i]) + 1), accuracy[i], label=labels[i])
    elif accuracy[i] and isinstance(accuracy[i][0], (list, np.ndarray)):
        plt.plot(range(1, len(accuracy[i][0]) + 1), accuracy[i][0], label=labels[i])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot comparison of loss
plt.figure(figsize=(10, 6))
plt.title('Loss Comparisons')
for i in range(len(losses)):
    if i == 2:
        if losses[i] and isinstance(losses[i][0], (list, np.ndarray)):
            plt.plot(range(1, len(losses[i][0]) + 1), losses[i][0], label=labels[i])
    elif i == 5:
        if losses[i] and isinstance(losses[i], (list, np.ndarray)):  # Corrected this line
            plt.plot(range(1, len(losses[i]) + 1), losses[i], label=labels[i])
    elif losses[i] and isinstance(losses[i][0], (list, np.ndarray)):
        plt.plot(range(1, len(losses[i][0]) + 1), losses[i][0], label=labels[i])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:

#Part 4 – Implementation of Participation Probability in FEDRETAIL Framework using NN 
#Participation Probability [0.09, 0.2, 0.7]
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the number of clients and local epochs
num_clients = 10
local_epochs = 2

# Shuffle the data before splitting
shuffled_indices = np.random.permutation(len(train_images))
shuffled_train_images = train_images[shuffled_indices]
shuffled_train_labels = train_labels[shuffled_indices]

# Split the shuffled data into equal partitions for each client
split_train_images = np.array_split(shuffled_train_images, num_clients)
split_train_labels = np.array_split(shuffled_train_labels, num_clients)

# Initialize global model for FEDRETAIL
global_model_fedretail = create_model()

# Initialize lists to store accuracy and losses for each scenario
accuracy = []
losses = []

# Centralized training
centralized_model = create_model()
history_centralized = centralized_model.fit(train_images, train_labels, epochs=150, 
                                            validation_data=(test_images, test_labels), verbose=1)
accuracy.append(history_centralized.history['val_accuracy'])
losses.append(history_centralized.history['loss'])

# FEDRETAIL Implementing (Horizontal FL) with different participation probabilities
participation_probabilities = [0.09, 0.2, 0.7]
for part_prob in participation_probabilities:
    part_prob_accuracy = []
    part_prob_losses = []
    for epoch in range(150):
        local_weights = []
        local_biases = []
        for i in range(num_clients):
            # Create and compile local model
            local_model = create_model()
            local_model.set_weights(global_model_fedretail.get_weights())

            # Train local model with participation probability
            if np.random.rand() < part_prob:
                local_model.fit(split_train_images[i], split_train_labels[i], epochs=local_epochs, verbose=0)

            # Get local model weights
            local_weights.append(local_model.layers[1].get_weights()[0])
            local_biases.append(local_model.layers[1].get_weights()[1])

        # Aggregate local weights
        global_weights = np.mean(local_weights, axis=0)
        global_biases = np.mean(local_biases, axis=0)

        # Update global model with aggregated weights
        global_model_fedretail.layers[1].set_weights([global_weights, global_biases])

        # Evaluate global model
        loss, accuracy_fedretail = global_model_fedretail.evaluate(test_images, test_labels, verbose=0)
        part_prob_accuracy.append(accuracy_fedretail)
        part_prob_losses.append(loss)
        print(f"FEDRETAIL with Part. Prob. {part_prob} - Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy_fedretail}")

    accuracy.append(part_prob_accuracy)
    losses.append(part_prob_losses)

# Plot comparison of accuracy
plt.figure(figsize=(10, 6))
plt.title('Accuracy Comparisons')
epochs = range(1, 151)
plt.plot(epochs, accuracy[0], label='Centralized FL')
plt.plot(epochs, accuracy[1], label='FEDRETAIL (part. prob. 0.09)')
plt.plot(epochs, accuracy[2], label='FEDRETAIL (part. prob. 0.2)')
plt.plot(epochs, accuracy[3], label='FEDRETAIL (part. prob. 0.7)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot comparison of loss
plt.figure(figsize=(10, 6))
plt.title('Loss Comparisons')
plt.plot(epochs, losses[0], label='Centralized FL')
plt.plot(epochs, losses[1], label='FEDRETAIL (part. prob. 0.09)')
plt.plot(epochs, losses[2], label='FEDRETAIL (part. prob. 0.2)')
plt.plot(epochs, losses[3], label='FEDRETAIL (part. prob. 0.7)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
