#
# Methods copied directly from
# https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/framework/optimizer_utils.py
# to add custom logic.
#

# Copyright 2018, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common building blocks for federated optimization algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six
from six.moves import zip
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils

nest = tf.contrib.framework.nest


def _create_optimizer_and_server_state(model, optimizer):
    """A helper for server computations that constructs the model and optimizer.

    This code is needed both in server_init (to introduce variables so
    we can read their initial values) and in server_update_model.

    Args:
      model: A `tff.learning.Model`.
      optimizer: A `tf.train.Optimizer`.

    Returns:
      A tuple of (apply_delta_fn, server_state), where:
        *  apply_delta_fn is a TensorFlow function that takes a model delta and
           updates the trainable model weights as well as possibly optimizer_state
           variables introduced by the optimizer.
        *  server_state is a `tff.learning.framework.ServerState` tuple holding
           those variables.
    """

    @tf.contrib.eager.defun(autograph=False)
    def apply_delta(delta):
        """Applies `delta` to `model.weights`."""
        nest.assert_same_structure(delta, model.weights.trainable)
        grads_and_vars = nest.map_structure(lambda x, v: (-1.0 * x, v),
                                            nest.flatten(delta),
                                            nest.flatten(model.weights.trainable))
        # N.B. This may create variables.
        optimizer.apply_gradients(grads_and_vars, name='server_update')
        return tf.constant(1)  # We have to return something.

    # Create a dummy input and trace apply_delta so that
    # we can determine the optimizer's variables.
    weights_delta = nest.map_structure(tf.zeros_like, model.weights.trainable)

    # TODO(b/109733734): We would like to call get_concrete_function,
    # but that does not currently work with structured inputs.
    # For now, we just call the function on dummy input, which
    # still ensures the function is traced (so variables are created).
    apply_delta(delta=weights_delta)

    # N.B. Using to_var_dict doesn't work here, because we
    # may get non-unique names, so we just use a flat list.
    optimizer_vars = optimizer.variables()

    return apply_delta, ServerState(
        model=model.weights, optimizer_state=optimizer_vars)


def server_update_model(current_server_state, weights_delta, model_fn,
                        optimizer_fn):
    """Updates `server_state` based on `weights_delta`.

    Args:
      current_server_state: A `tff.learning.framework.ServerState` namedtuple.
      weights_delta: An update to the trainable variables of the model.
      model_fn: A no-arg function that returns a `tff.learning.Model`. Passing in
        a function ensures any variables are created when server_update_model is
        called, so they can be captured in a specific graph or other context.
      optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`. As with
        model_fn, we pass in a function to control when variables are created.

    Returns:
      An updated `tff.learning.framework.ServerState`.
    """
    py_typecheck.check_type(current_server_state, ServerState)
    py_typecheck.check_type(weights_delta, collections.OrderedDict)
    model = model_utils.enhance(model_fn())
    optimizer = optimizer_fn()
    apply_delta_fn, server_state_vars = _create_optimizer_and_server_state(
        model, optimizer)

    # We might have a NaN value e.g. if all of the clients processed
    # had no data, so the denominator in the federated_mean is zero.
    # If we see any NaNs, zero out the whole update.
    no_nan_weights_delta, _ = tensor_utils.zero_all_if_any_non_finite(
        weights_delta)

    # TODO(b/124538167): We should increment a server counter to
    # track the fact a non-finite weiths_delta was encountered.

    @tf.contrib.eager.function(autograph=False)
    def update_model_inner():
        """Applies the update."""
        nest.map_structure(tf.assign, server_state_vars, current_server_state)
        apply_delta_fn(no_nan_weights_delta)
        return server_state_vars

    return update_model_inner()
