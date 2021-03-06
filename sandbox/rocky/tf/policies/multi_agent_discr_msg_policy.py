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


class MultiAgentDiscreteMsgMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            n_row,
            n_col,
            n_agent,
            env_spec,
            feature_dim = 16, # feature from each agent's local information
            msg_dim = 4, # when msg_dim == 0, no communication
            conv_layers=[4,4,4],
            hidden_layers=[],
            comm_layers = [10],
            act_dim = 5,
            shared_weights = False, # whether agents share the weights
            hid_func = tf.nn.elu
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """

        assert(shared_weights == False) # do not support shared weights
        assert(msg_dim == 0 or msg_dim == n_agent)

        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Product)
        assert(n_row > 0 and n_col > 0 and n_agent > 0)
        if (n_agent == 1):
            assert(msg_dim == 0)

        self.shared_weights = shared_weights
        self.n_row, self.n_col, self.n_agent = n_row, n_col, n_agent
        self.msg_dim = msg_dim
        self.act_dim = act_dim

        # NOTE: *IMPORTANT* 
        #    RLLab requires all the passed-in arguments to remain unchanged!
        #    Otherwise error will happen when loading a saved model
        self.hidden_layers = hidden_layers.copy()

        map_size = n_row * n_col

        with tf.variable_scope(name):

            self.input = L.InputLayer((None, env_spec.observation_space.flat_dim))

            shared_map = L.SliceLayer(self.input,
                                      indices=slice(map_size * (2 * n_agent + 1)),
                                      axis = 1)
            outputs = []
            msgs = []
            msg_probs = []

            with tf.variable_scope(name + '-single-net') as scope:
                if msg_dim == 0:
                    # no communication, directly output probabilities
                    out_dim = act_dim
                    out_nonlinear = tf.nn.softmax
                    if feature_dim > 0:
                        self.hidden_layers.append(feature_dim)
                else:
                    # communication
                    out_dim = feature_dim + msg_dim
                    out_nonlinear = hid_func
                # construct network
                for i in range(n_agent):
                    cur_self_map = L.SliceLayer(self.input,
                                                indices=slice(map_size*(2*n_agent+1+i),map_size*(2*n_agent+2+i)),
                                                axis = 1)
                    # map, agent_locations, shuffled agent_goals, obs goal
                    n_channel = 2 * n_agent + 2
                    comb_input = L.concat([shared_map, cur_self_map], axis=1)
                    recons_input = L.reshape(comb_input, ([0], n_channel, map_size)) #(batch, 2*n_agent+2, map_size)
                    cur_input = L.reshape(
                                    L.dimshuffle(recons_input, [0, 2, 1]),
                                    ([0],n_row,n_col,n_channel)
                                ) # (batch, flat_dim(row, col, channel))

                    curr_name = "single_conv_network"
                    if not self.shared_weights:
                        curr_name += "_%d" % (i)
                    single_network = ConvNetwork(
                        name = curr_name,
                        input_shape=(n_row,n_col,n_channel),
                        output_dim=out_dim,
                        conv_filters=conv_layers,
                        conv_filter_sizes = [3] * len(conv_layers),
                        conv_strides = [1] * len(conv_layers),
                        conv_pads = ['SAME'] * len(conv_layers),
                        hidden_nonlinearity = hid_func,
                        hidden_sizes = self.hidden_layers,
                        output_nonlinearity = out_nonlinear,
                        input_layer = cur_input
                    )

                    cur_output = single_network.output_layer # (batch, out_dim)
                    if (msg_dim == 0):
                        outputs.append(cur_output)
                    else:
                        msg_logits = L.SliceLayer(cur_output,
                                                  indices=slice(feature_dim,None),
                                                  axis=1)
                        msg_p = L.OpLayer(msg_logits, tf.nn.softmax)
                        msg_probs.append(msg_p)
                        func = lambda x: tf.one_hot(
                                    tf.reshape(tf.multinomial(x,1),[-1]),
                                    self.n_agent,on_value=1.0,dtype=tf.float32)
                        msg_a = L.OpLayer(msg_p, func)
                        msgs.append(msg_a)
                        outputs.append(
                            L.SliceLayer(cur_output,
                                         indices=slice(feature_dim),
                                         axis=1)
                        )
                    if i == 0 and self.shared_weights:
                        scope.reuse_variables()

            final_outputs = [] # list of actions probs by each agent
            if msg_dim == 0:
                final_outputs = outputs
            else:
                with tf.variable_scope(name+'-comm-net') as scope:
                    for i in range(self.n_agent):
                        # compute incomming messages
                        income_msgs = [msgs[j] for j in range(self.n_agent) if j != i]
                        comb_features = L.concat([outputs[i]]+income_msgs, axis = 1)
                        comb_dim = feature_dim + msg_dim * len(income_msgs)
                        # communication network
                        curr_name = 'single-comm-MLP'
                        if not self.shared_weights:
                            curr_name += "_%d" % (i)
                        comm_network = MLP(
                            name = curr_name,
                            input_shape = (comb_dim,),
                            output_dim = act_dim,
                            hidden_sizes = comm_layers,
                            hidden_nonlinearity=hid_func,
                            output_nonlinearity=tf.nn.softmax,
                            input_layer=comb_features
                        )

                        final_outputs.append(comm_network.output_layer)

                        if i == 0 and self.shared_weights:
                            scope.reuse_variables()

            if msg_dim == 0:
                assert(len(msg_probs) == 0)
            self.output_probs = L.concat(final_outputs+msg_probs, axis = 1)
            # gen output function
            output_vars = L.get_output(self.output_probs)
            if msg_dim > 0:
                # also output comm msgs
                output_vars = [output_vars]
                for m in msgs:
                    # [TODO] switch to .output_var????
                    output_vars.append(L.get_output(m))
            self.f_prob = tensor_utils.compile_function(
                [self.input.input_var],
                output_vars
            )

            self._dist = ProductDistribution([Categorical(c.n) for c in \
                                              env_spec.action_space.components])

            super(MultiAgentDiscreteMsgMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, final_outputs)

    def check_length(self, outputs):
        l = len(outputs)
        if self.msg_dim > 0:
            assert(l == self.n_agent * 2)
        else:
            assert(l == self.n_agent)

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        val = L.get_output(self.output_probs, {self.input: tf.cast(obs_var, tf.float32)})
        outputs = self._dist._split_x(val) # split compact outputs
        self.check_length(outputs) # assert
        D = dict()
        for i, out in enumerate(outputs):
            D['id_%d_prob' % i] = out
        return D

    @overrides
    def dist_info(self, obs, state_infos=None):
        val = self.f_prob(obs)
        if self.msg_dim > 0:
            val = val[0]
        outputs = self._dist._split_x(val) # split compact outputs
        self.check_length(outputs) # assert
        D = dict()
        for i, out in enumerate(outputs):
            D['id_%d_prob' % i] = out
        return D

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        val = self.f_prob([flat_obs])
        msgs = None
        if self.msg_dim > 0:
            msgs = val[1:]
            val = val[0]
        prob_arr = self._dist._split_x(val)
        probs = [p[0] for p in prob_arr]
        actions = []
        for i,s in enumerate(self.action_space.components):
            p = probs[i] if i < self.n_agent or msgs is None else msgs[i-self.n_agent]
            actions.append(s.weighted_sample(p))
        #actions = [s.weighted_sample(p) for s, p in zip(self.action_space.components, probs)]
        D = dict()
        for i, p in enumerate(probs):
            D['id_%d_prob' % i] = p
        if msgs is not None:
            for i in range(self.n_agent):
                D['msg_%d' % i] = msgs[i]
        return actions, D

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        val = self.f_prob(flat_obs)
        msgs = None
        if self.msg_dim > 0:
            msgs = val[1:]
            val = val[0]
        probs = self._dist._split_x(val)
        self.check_length(probs) # assert
        rev_actions = []
        for i, c in enumerate(self.action_space.components):
            p = probs[i] if i < self.n_agent or msgs is None else msgs[i-self.n_agent]
            rev_actions.append(list(map(c.weighted_sample, p)))
        #rev_actions = [list(map(c.weighted_sample, p)) for p,c in zip(probs,self.action_space.components)]
        n = len(rev_actions)
        self.check_length(rev_actions) # assert
        batch_size = len(rev_actions[0])
        actions = [[rev_actions[i][b] for i in range(n)] for b in range(batch_size)] #[batch_size, n_agent]
        D = dict()
        for i, p in enumerate(probs):
            D['id_%d_prob' % i] = p
        if msgs is not None:
            for i in range(self.n_agent):
                D['msg_%d' % i] = msgs[i]
        return actions, D

    @property
    def distribution(self):
        return self._dist
