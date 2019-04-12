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
# Number of time instances to run the simulation for before evaluation
SIMULATION_TIME = 10


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

        # Run simulation
        for t in range(1, SIMULATION_TIME + 1):
            print("t: " + str(t))
            for n in nodes:
                n.move()
                n.learn()

            for n1 in nodes:
                for n2 in nodes:
                    if n1 != n2 and n1.sees(n2):
                        print(str(n1.identifier) + " sees " + str(n2.identifier))
                        n1.share(n2)

        for n in nodes:
            n.evaluate()

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
