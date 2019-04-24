from __future__ import absolute_import, division, print_function

import collections
import math
import os
import time

from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff
import random

from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils

from MnistModel import MnistModel, MnistTrainableModel
from federated.optimizer_utils import server_update_model

nest = tf.contrib.framework.nest

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

from Node import Node

# Maximum number of nodes (the dataset can have upwards of 1000s of clients, and thus 1000s of nodes would be created)
NODE_LIMIT = 10  # 100
# Number of rounds that should occur @SERVER
SIMULATION_NUM_ROUNDS = 3
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

        # Create directory to store results
        experiment_storage = "./storage/experiments/{}/".format(time.time())
        os.makedirs(experiment_storage)

        # File storing metrics at each coordinator node
        f_coordinator = open(experiment_storage + "coordinator_metrics.csv", "a")
        f_coordinator.write("coordinator_id,round_num,num_neighbors,accuracy,loss\n")

        # File storing metrics on our global model
        f_global = open(experiment_storage + "global_metrics.csv", "a")
        f_global.write("round_num,accuracy,loss\n")

        # File storing metrics on the standard federated learning
        f_global_regular = open(experiment_storage + "global_metrics_regular.csv", "a")
        f_global_regular.write("round_num,accuracy,loss\n")

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

        test_node_ids = test.client_ids
        federated_test_data = Simulation.make_federated_data(test, test_node_ids)

        # Create model graph
        model_fn = MnistTrainableModel

        # Create global-model state (model at the server)
        evaluation = tff.learning.build_federated_evaluation(MnistModel)
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        iterative_process_regular = tff.learning.build_federated_averaging_process(model_fn)
        global_state = iterative_process.initialize()
        global_state_regular = iterative_process.initialize()

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
                    states[c], metrics = Simulation.fed_learn(train, nodes_seen_by_coordinator[c], states[c],
                                                              iterative_processes[c])
                    print('coordinator: {} round {:2d}, metrics={}'.format(c.identifier, round_num, metrics))
                    f_coordinator.write("{},{},{}\n".format(round_num,
                                                            metrics.accuracy,
                                                            metrics.loss))

            # Determine the weights delta for the round from the coordinators compared to the SERVER.
            client_weights_deltas = []
            for s in states:
                state = states[s]
                values = []
                for x in state.model.trainable:
                    values.append(x)

                weights_delta = collections.OrderedDict({
                    "weights": values[0],
                    "bias": values[1],
                })

                client_weights_deltas.append(weights_delta)

            # Now that the round is completed, we share the models from the coordinators to the SERVER for averaging.
            weights_per_coordinator = []
            bias_per_coordinator = []
            for c_i in client_weights_deltas:
                weights_per_coordinator.append(c_i['weights'])
                bias_per_coordinator.append(c_i['bias'])
            new_weights_deltas = tf.math.reduce_mean(weights_per_coordinator, 0).numpy()
            new_bias_deltas = tf.math.reduce_mean(bias_per_coordinator, 0).numpy()

            # Bad way to set these variables, but it works for a proof of concept at least.
            for i in range(0,len(new_weights_deltas)):
                global_state.model.trainable.weights[i] = new_weights_deltas[i]
            for i in range(0, len(new_bias_deltas)):
                global_state.model.trainable.bias[i] = new_bias_deltas[i]

            # # Evaluate Global Model (our method)
            # metrics = evaluation(global_state.model, federated_test_data)
            # print(metrics)
            # f_global.write("{},{},{}\n".format(round_num,
            #                                    metrics.accuracy,
            #                                    metrics.loss))
            #
            # # Evaluate Global Model (regular federated learning on coordinator nodes)
            # metrics = evaluation(global_state_regular.model, federated_test_data)
            # print(metrics)
            # f_global.write("{},{},{}\n".format(round_num,
            #                                    metrics.accuracy,
            #                                    metrics.loss))

        print(type(global_state))
        print(global_state)

        f_coordinator.close()
        f_global.close()
        f_global_regular.close()

        end = time.time()
        time_taken = end - start
        print("Simulation took: {}min {}s".format(math.floor(time_taken / 60), time_taken % 60))

    @staticmethod
    def fed_learn(train, nodes, state, iterative_process):
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


if __name__ == '__main__':
    Simulation.main()
