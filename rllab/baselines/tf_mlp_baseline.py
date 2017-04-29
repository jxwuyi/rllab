import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides

import tensorflow as tf
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors


def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

def dense(x, size, name, weight_init=None, bias_init=None, return_vars = True):
    """
    Dense (fully connected) layer
    """
    if isinstance(weight_init, float):
        weight_init = normc_initializer(weight_init)
    if weight_init is None:
        weight_init = tf.contrib.layers.xavier_initializer()
    if bias_init is None:
        bias_init = tf.constant_initializer(0, dtype=tf.float32)
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [x.get_shape()[1], size], initializer=weight_init)
        b = tf.get_variable('bias', [1, size], initializer=bias_init)
        ret = tf.matmul(x, w) + b
    if return_vars:
        return ret, w, b
    return ret

class TFMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            hidden_layers=(64, 32)
    ):
        Serializable.quick_init(self, locals())
        super(TFMLPBaseline, self).__init__(env_spec)
        self.net = None
        self.layers = tuple(hidden_layers)
        if self.layers[-1] != 1:
            self.layers += (1,)
        self.var_shapes = []
        self.vars = None
        self.input_dim = env_spec.observation_space.flat_dim
        self.init_vals = None
        self.init_flag = False
        for i, d in enumerate(hidden_layers):
            prev_d = self.input_dim if i == 0 else hidden_layers[i-1]
            self.var_shapes.append((prev_d, d))  # proj matrix
            self.var_shapes.append((1, d))  # bias

    def create_net(self):
        with tf.variable_scope('VF'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None], name="y")
            self.vars = []
            prev = self.x
            for l, d in enumerate(self.layers):
                if self.init_vals is None:
                    init_w = init_b = None
                else:
                    init_w = self.init_vals[2 * l]
                    init_b = self.init_vals[2 * l + 1]
                hidden, w, b = dense(prev, d, 'hidden_{}'.format(l), weight_init=init_w, bias_init=init_b, return_vars=True)
                self.vars.append(w)
                self.vars.append(b)
                if d > 1:
                    prev = tf.nn.relu(hidden)
                else:  # last layer
                    self.net = hidden
            self.net = tf.reshape(self.net, (-1,))
            l2 = (self.net - self.y) * (self.net - self.y)
            self.train = tf.train.AdamOptimizer().minimize(l2)
            self.session = tf.get_default_session()
            self.session.run(tf.initialize_variables(self.vars))

    @overrides
    def fit(self, paths):
        self.init_flag = True
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.reshape(np.concatenate([p["returns"] for p in paths]), [-1])  # ensure shape
        if self.net is None:
            self.create_net()
        for _ in range(50):
            self.session.run(self.train, {self.x: observations, self.y: returns})

    @overrides
    def predict(self, path):
        if not self.init_flag:
            if self.net is None:
                self.create_net()
            return np.zeros(len(path["rewards"]))  # reward
        else:
            ret = self.session.run(self.net, {self.x: path["observations"]})
            return np.reshape(ret, (ret.shape[0],))

    @overrides
    def get_param_values(self, **tags):
        return flatten_tensors([tf.get_default_session().run(var) for var in self.vars])

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self.init_vals = unflatten_tensors(flattened_params, self.var_shapes)
        self.create_net()
