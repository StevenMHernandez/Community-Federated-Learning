from __future__ import absolute_import, division, print_function

import collections
import math
import time

from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff
import random

nest = tf.contrib.framework.nest

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

from Node import Node

# Maximum number of nodes (the dataset can have upwards of 1000s of clients, and thus 1000s of nodes would be created)
NODE_LIMIT = 2
# Number of rounds that should occur @SERVER
SIMULATION_NUM_ROUNDS = 10
# Number of time instances that should occur per round
SIMULATION_NUM_T_PER_ROUND = 10
# Number of coordinator nodes to select per round
NUM_COORDINATOR_NODES = 1


class Simulation:
    @staticmethod
    def main():
        start = time.time()

        # Initialize nodes based on the given dataset
        nodes = []
        train, test = tff.simulation.datasets.emnist.load_data(True, 'storage')
        node_ids = train.client_ids
        if len(node_ids) > NODE_LIMIT:
            random.shuffle(node_ids)
            node_ids = node_ids[:NODE_LIMIT]
        for n_i in node_ids:
            nodes.append(Node(n_i))
        sample_batch = Simulation.create_sample_batch(train)

        # Create model graph
        def model_fn():
            keras_model = Simulation.create_compiled_keras_model()
            return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

        # Create global-model state (model at the server)
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        global_state = iterative_process.initialize()

        # Run simulation a given number of rounds
        for round_num in range(1, SIMULATION_NUM_ROUNDS + 1):
            # Select nodes to be coordinators
            coordinators = nodes[0:NUM_COORDINATOR_NODES]

            # Initiate the coordinator's state (local-modal) based on the global state (global-state)
            iterative_processes = {}
            states = {}
            for c in coordinators:
                iterative_processes[c] = iterative_process
                states[c] = global_state

            # Run simulation for SIMULATION_NUM_T_PER_ROUND times instances
            # noting which nodes were seen by coordinators
            nodes_seen_by_coordinator = {}
            for t in range(1, SIMULATION_NUM_T_PER_ROUND + 1):
                # Move all nodes once
                for n in nodes:
                    n.move()

                # Check if any nodes have been seen by the coordinators
                for c in coordinators:
                    for n in nodes:
                        if c != n and c.sees(n):
                            if c not in nodes_seen_by_coordinator:
                                nodes_seen_by_coordinator[c] = []
                            if n not in nodes_seen_by_coordinator[c]:
                                nodes_seen_by_coordinator[c].append(n)

            # For all nodes seen by coordinator for a given round
            # learn together
            for c in coordinators:
                if c in nodes_seen_by_coordinator:
                    print("Round {}, coordinator {} learns from [{}]".format(round_num, c.identifier, ",".join(
                        map(lambda x: x.identifier, nodes_seen_by_coordinator[c]))))
                    states[c], metrics = Simulation.fl_c_to_n(train, nodes_seen_by_coordinator[c], states[c],
                                                              iterative_processes[c])
                    print('coordinator: {} round {:2d}, metrics={}'.format(c.identifier, round_num, metrics))

            # Now that the round is completed, we share the models from the coordinators to the SERVER for averaging.
            # TODO: federated learning (SERVER from coordinators)
            global_state = states[coordinators[0]]

        print(type(global_state))
        print(global_state)

        # TODO: Evaluation
        end = time.time()
        time_taken = end - start
        print("Simulation took: {}min {}s".format(math.floor(time_taken / 60), time_taken % 60))

    @staticmethod
    def fl_c_to_n(train, nodes, state, iterative_process):
        """
        Run Federated Learning between Coordinators to mobile Nodes

        :param train:
        :param nodes:
        :param state:
        :param iterative_process:
        :return:
        """
        node_ids = map(lambda x: x.identifier, nodes)
        federated_train_data = Simulation.make_federated_data(train, node_ids)
        return iterative_process.next(state, federated_train_data)

    @staticmethod
    def preprocess(dataset):
        NUM_EPOCHS = 20
        BATCH_SIZE = 10
        SHUFFLE_BUFFER = 500

        def element_fn(element):
            return collections.OrderedDict([
                ('x', tf.reshape(element['pixels'], [-1])),
                ('y', tf.reshape(element['label'], [1])),
            ])

        return dataset.repeat(NUM_EPOCHS).map(element_fn) \
            .shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

    @staticmethod
    def make_federated_data(client_data, client_ids):
        return [Simulation.preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

    @staticmethod
    def create_sample_batch(train):
        example_dataset = train.create_tf_dataset_for_client(
            train.client_ids[0])

        preprocessed_example_dataset = Simulation.preprocess(example_dataset)

        return nest.map_structure(lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

    @staticmethod
    def create_compiled_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                  kernel_initializer='zeros', input_shape=(784,))
        ])

        def loss_fn(y_true, y_pred):
            return tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

        model.compile(
            loss=loss_fn,
            optimizer=gradient_descent.SGD(learning_rate=0.02),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model


if __name__ == '__main__':
    Simulation.main()
