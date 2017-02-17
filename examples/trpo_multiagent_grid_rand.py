from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.multi_agent_grid_world_guided_env import MultiAgentGridWorldGuidedEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.multi_agent_categorical_policy import MultiAgentCategoricalMLPPolicy
import tensorflow as tf

# store policy
import rllab.misc.logger as logger

# arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=501)
parser.add_argument('--pathlen', type=int, default=30)
parser.add_argument('--logdir', type=str, default='None',
                   help='repo name for snapshots. the repo is under ../temp/ dir.')
parser.add_argument('--msgdim', type=int, default=0,
                   help='msg dimension. 0 for no communication')
parser.add_argument('--sharewei', dest='sharewei', action='store_true')
parser.set_defaults(sharewei=False)
parser.add_argument('--swapobs', dest='swapobs', action='store_true')
parser.set_defaults(swapobs=False)
parser.add_argument('--activation', choices=['relu','elu'], default='elu')
args = parser.parse_args()
##

policy_dir = None if args.logdir == 'None' else '../temp/rand/'+args.logdir
if policy_dir is not None:
    import os
    if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
store_mode = None if policy_dir is None else 'gap'
store_gap = 10

map_desc = '4x4-empty' # map description, see multi_agent_grid_world_env.py
n_row = 4  # n_row and n_col need to be compatible with desc
n_col = 4
n_agent = 2 # 2 <= agents <= 6

logger.set_snapshot_dir(policy_dir)
logger.set_snapshot_mode(store_mode)
logger.set_snapshot_gap(store_gap)

env = TfEnv(normalize(MultiAgentGridWorldGuidedEnv(n = n_agent,
                                                   desc = map_desc,
                                                   collision = True,
                                                   swap_goal_obs = args.swapobs)))

policy = MultiAgentCategoricalMLPPolicy(
    'MAP',
    n_row,
    n_col,
    n_agent,
    env_spec=env.spec,
    feature_dim = 10, # feature from each agent's local information
    msg_dim = args.msgdim, # when msgdim == 0, no communication
    conv_layers=[], # number of conv-layers and the number of kernels
    hidden_layers=[64, 32, 20], # hidden layers after conv-layers
    comm_layers = [10], # hidden layers after receiving msgs from other agents
    act_dim = 5, # always 5 in grid domain: 4 directions + stay
    shared_weights = args.sharewei,
    hid_func = tf.nn.elu if args.activation is 'elu' else tf.nn.relu
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=20000,
    max_path_length=args.pathlen,
    n_itr=args.iter,
    discount=0.99,
    step_size=0.1,
)
"""
algo = VPG(
    env = env,
    policy = policy,
    baseline = baseline,
    batch_size = 3000,
    max_path_length=20,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
"""
algo.train()
