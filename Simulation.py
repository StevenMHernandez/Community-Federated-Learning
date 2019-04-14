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

from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils

from federated.optimizer_utils import server_update_model

nest = tf.contrib.framework.nest

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

from Node import Node

# Maximum number of nodes (the dataset can have upwards of 1000s of clients, and thus 1000s of nodes would be created)
NODE_LIMIT = 20
# Number of rounds that should occur @SERVER
SIMULATION_NUM_ROUNDS = 10
# Number of time instances that should occur per round
SIMULATION_NUM_T_PER_ROUND = 10
# Number of coordinator nodes to select per round
NUM_COORDINATOR_NODES = 2

#
# Node Movement Variables
#
REGION_WIDTH = 100
REGION_HEIGHT = 100
TRANSMISSION_RADIUS = 10
MIN_SPEED = 0.1
MAX_SPEED = 1
MIN_PAUSE = 1
MAX_PAUSE = 5
MIN_TRAVEL = 1
MAX_TRAVEL = SIMULATION_NUM_ROUNDS * SIMULATION_NUM_T_PER_ROUND


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
            nodes.append(
                Node(n_i, TRANSMISSION_RADIUS, REGION_HEIGHT, REGION_WIDTH, MIN_SPEED, MAX_SPEED, MIN_PAUSE, MAX_PAUSE,
                     MIN_TRAVEL, MAX_TRAVEL))
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
            # The coordinator is always seen so that learning occurs at a minimum for the coordinator node's dataset
            for c in coordinators:
                nodes_seen_by_coordinator[c] = [c]
            for t in range(1, SIMULATION_NUM_T_PER_ROUND + 1):
                # Move all nodes once
                for n in nodes:
                    n.move()
                    # print(n.identifier, n.x, n.y)

                # Check if any nodes have been seen by the coordinators
                for c in coordinators:
                    for n in nodes:
                        if c != n and c.sees(n):
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
            for s in states:
                state = states[s]
                values = []
                for x in state.model.trainable:
                    values.append(x)

                weights_delta = collections.OrderedDict({
                    "dense/kernel": values[0],
                    "dense/bias": values[1],
                })

                server_state = optimizer_utils.ServerState(
                    model=model_utils.ModelWeights.from_tff_value(state.model),  # should be global state
                    optimizer_state=list(state.optimizer_state))

                global_state = server_update_model(
                    server_state,
                    weights_delta,
                    model_fn=model_fn,
                    optimizer_fn=lambda: gradient_descent.SGD(learning_rate=1.0))

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
