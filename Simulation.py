from __future__ import absolute_import, division, print_function

import collections
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

# Maximum number of nodes (datasets can have upwards of 1000s of clients, and thus 1000s of nodes would be created)
NODE_LIMIT = 25
# Number of rounds that should occur @SERVER
SIMULATION_NUM_ROUNDS = 10
# Number of time instances that should occur per round
SIMULATION_NUM_T_PER_ROUND = 10
# Number of coordinator nodes to select per round
NUM_COORDINATOR_NODES = 4


class Simulation:
    @staticmethod
    def main():
        # Initialize nodes based on the given dataset
        nodes = []
        train, test = tff.simulation.datasets.emnist.load_data(True, 'storage')
        node_ids = train.client_ids
        if len(node_ids) > NODE_LIMIT:
            random.shuffle(node_ids)
            node_ids = node_ids[:NODE_LIMIT]
        for n_i in node_ids:
            nodes.append(Node(n_i))

        # Run simulation a given number of rounds
        for round_num in range(1, SIMULATION_NUM_ROUNDS + 1):
            # Select random nodes to be coordinators
            coordinators = nodes[0:NUM_COORDINATOR_NODES]

            nodes_seen_by_coordinator = {}

            for t in range(1, SIMULATION_NUM_T_PER_ROUND + 1):
                # Move all nodes once
                for n in nodes:
                    n.move()

                # Check if any nodes have been seen by the coordinators
                for c in coordinators:
                    for n in nodes:
                        if c != n and c.sees(n):
                            print(str(c.identifier) + " sees " + str(n.identifier))
                            if c not in nodes_seen_by_coordinator:
                                nodes_seen_by_coordinator[c] = []
                            nodes_seen_by_coordinator[c].append(n)

            # For all nodes seen by coordinator for a given round
            # learn together
            for c in coordinators:
                if c in nodes_seen_by_coordinator:
                    print("Round {}, coordinator {} learns from [{}]".format(round_num, c.identifier, ",".join(map(lambda x: x.identifier, nodes_seen_by_coordinator[c]))))
                    for n in nodes_seen_by_coordinator[c]:
                        # TODO: federated learning (coordinator from mobile nodes)
                        pass

            # Now that the round is completed, we share the models from the coordinators to the SERVER for averaging.
            # TODO: federated learning (SERVER from coordinators)

        # TODO: Evaluation

    @staticmethod
    def preprocess(dataset):
        NUM_EPOCHS = 5
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
