from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP, ConvNetwork
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.distributions.product_distribution import ProductDistribution
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces import Discrete, Box, Product
import tensorflow as tf


class SingleAgentDiscreteMsgMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            n_row,
            n_col,
            n_goals,
            env_spec,
            conv_layers=(32, 16, 16),
            hidden_layers=(32,),
            act_dim=5,
            hid_func=tf.nn.relu,
            opt_msg=False,
    ):
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)
        assert(n_row > 0 and n_col > 0 and n_goals > 1)

        self.opt_msg = opt_msg
        self.n_row, self.n_col, self.n_goals = n_row, n_col, n_goals
        self.msg_dim = msg_dim = n_goals
        self.act_dim = act_dim

        # NOTE: *IMPORTANT* 
        #    RLLab requires all the passed-in arguments to remain unchanged!
        #    Otherwise error will happen when loading a saved model
        self.hidden_layers = hidden_layers.copy()

        map_size = n_row * n_col

        with tf.variable_scope(name):

            # observation state:
            #  ch#0: global map
            #  ch#1~n_goals: goals
            #  ch#n_goals+1: agent location
            #  ch#n_goals+2: agent observation
            self.input = L.InputLayer((None, env_spec.observation_space.flat_dim))

            shared_map = L.SliceLayer(self.input,
                                      indices=slice(map_size * (n_goals + 1)),
                                      axis=1)
            agent_loc = L.SliceLayer(self.input,
                                     indices=slice(map_size * (n_goals + 1), map_size * (n_goals+2)),
                                     axis=1)
            agent_goal = L.SliceLayer(self.input,
                                      indices=slice(map_size * (n_goals + 2), map_size * (n_goals + 3)),
                                      axis=1)
            msg = None
            with tf.variable_scope(name+'/message-net') as scope:
                comb_input = L.concat([shared_map, agent_goal], axis=1)
                recons_input = L.reshape(comb_input, ([0], n_goals+2, map_size))  # (batch, n_goals+2, map_size)
                if opt_msg:  # output the optimal message
                    def opt_func(x):  # compute the optimal message, deterministic
                        """
                        :param x: [batch, n_cn_goals + 2, map_size]
                            ch#0: global map
                            ch#1~n_goals: goals
                            ch#n_goals+1: observation
                        :return: [batch, msg_dim]
                        """
                        goals = x[:, 1:(n_goals)+1, :]  # [batch, n_goals/msg_dim, map_size]
                        obs = x[:, n_goals+1:, :]
                        msg = tf.reduce_sum(goals * obs, axis=2)
                        return msg
                    msg = L.OpLayer(recons_input, opt_func, shape_op=lambda x: (x[0], msg_dim))
                else:
                    n_channel = n_goals + 2
                    cur_input = L.reshape(
                        L.dimshuffle(recons_input, [0, 2, 1]), ([0], n_row, n_col, n_channel)
                    )  # (batch, row, col, channel)
                    message_network = ConvNetwork(
                        name='msg-conv-net',
                        input_shape=(n_row, n_col, n_channel),
                        output_dim=msg_dim,
                        conv_filters=conv_layers,
                        conv_filter_sizes=[3] * len(conv_layers),
                        conv_strides=[1] * len(conv_layers),
                        conv_pads=['SAME'] * len(conv_layers),
                        hidden_nonlinearity=hid_func,
                        hidden_sizes=self.hidden_layers,
                        output_nonlinearity=None,  # output logits
                        input_layer=cur_input
                    )
                    cur_output = message_network.output_layer  # (batch, msg_dim)
                    def sample_func(x):
                        """
                        :param x: [batch, msg_dim] logits for messages
                        :return: [batch, msg_dim] one-hot vector as sampled message, gradients computed by log(prob)
                        """
                        sample = tf.reshape(tf.multinomial(x, 1), [-1])
                        msg = tf.stop_gradient(tf.one_hot(sample, msg_dim))
                        log_prob = tf.nn.log_softmax(x) * msg
                        ret_msg = tf.stop_gradient(msg - log_prob) + log_prob  # return the gradient of log(prob)
                        return ret_msg
                    msg = L.OpLayer(cur_output, sample_func)

            with tf.variable_scope(name + '/policy-net') as scope:
                def tile_func(msg):
                    """
                    :param msg:  [batch, msg_dim/n_goals]
                    :return:  [batch, n_goals, map_size]
                    """
                    val = tf.reshape(msg, [-1, msg_dim, 1])
                    return tf.tile(val, [1, 1, map_size])
                msg_input = L.OpLayer(msg, tile_func, shape_op=lambda x: (x[0], n_goals, map_size))
                comb_input = L.concat([shared_map, msg_input, agent_loc], axis=1)
                n_channel = n_goals * 2 + 2  # gloabl map, obs_goals, msg_channels, cur loc
                recons_input = L.reshape(comb_input, ([0], n_channel, map_size))  # (batch, n_goals+2, map_size)
                cur_input = L.reshape(
                    L.dimshuffle(recons_input, [0, 2, 1]), ([0], n_row, n_col, n_channel)
                )  # (batch, row, col, channel)
                action_network = ConvNetwork(
                    name='action-conv-net',
                    input_shape=(n_row, n_col, n_channel),
                    output_dim=act_dim,
                    conv_filters=conv_layers,
                    conv_filter_sizes=[3] * len(conv_layers),
                    conv_strides=[1] * len(conv_layers),
                    conv_pads=['SAME'] * len(conv_layers),
                    hidden_nonlinearity=hid_func,
                    hidden_sizes=self.hidden_layers,
                    output_nonlinearity=tf.nn.softmax,
                    input_layer=cur_input
                )
                output_prob = action_network.output_layer  # (batch, msg_dim)

            self.output_prob = output_prob
            # gen output function
            output_vars = L.get_output(self.output_prob)
            self.f_prob = tensor_utils.compile_function(
                [self.input.input_var],
                output_vars
            )

            assert(env_spec.action_space.n == act_dim)
            self._dist = Categorical(env_spec.action_space.n)

            super(SingleAgentDiscreteMsgMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self,  [action_network.output_layer])

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        return dict(prob=L.get_output(self.output_prob, {self.input: tf.cast(obs_var, tf.float32)}))

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self.f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self.f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self.f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist
