from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.tf_mlp_baseline import TFMLPBaseline
from rllab.envs.single_agent_discr_comm_grid_world_guided_env import SingleAgentDiscrCommGridWorldGuidedEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.single_agent_discr_msg_policy import SingleAgentDiscreteMsgMLPPolicy
import tensorflow as tf

# store policy
import rllab.misc.logger as logger

# arguments
import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str)
parser.add_argument('--iter', type=int, default=501)
parser.add_argument('--pathlen', type=int, default=20)
parser.add_argument('--maxsteps', type=int, default=10000)
parser.add_argument('--maxkl', type=float, default=0.01)
parser.add_argument('--n_goal', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--logdir', type=str,
                   help='repo name for snapshots. the repo is under ../temp/ dir.')
parser.add_argument('--opt-msg', dest='opt_msg', action='store_true',
                   help='when this flag set, the speaker will always produce optimal messages')
parser.set_defaults(opt_msg=False)
parser.add_argument('--agent-escape', dest='escape', action='store_true',
                   help='when this flag set, agent will escape from the map immediately after reaching the goal')
parser.set_defaults(escape=False)
parser.add_argument('--activation', choices=['relu','elu'], default='relu')
parser.add_argument('--baseline', choices=['mlp','linear'], default='mlp')
parser.add_argument('--store_gap', type=int, default=20)
args = parser.parse_args()
##

prefix = '../temp/single_discr_comm/'
policy_dir = None if args.logdir is None else prefix+args.logdir
if policy_dir is not None:
    if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
store_mode = None if policy_dir is None else 'gap'
store_gap = args.store_gap

map_desc = '4x4-empty' # map description, see multi_agent_grid_world_env.py
n_row = 4  # n_row and n_col need to be compatible with desc
n_col = 4
n_goals = args.n_goal

if args.desc is not None:
    map_desc = args.desc

logger.set_snapshot_dir(policy_dir)
logger.set_snapshot_mode(store_mode)
logger.set_snapshot_gap(store_gap)

if policy_dir is not None:
    track_files = ['progress.csv','params.json','debug.log']
    for track_file in track_files:
        fname = policy_dir + '/' + track_file
        if os.path.isfile(fname):
            os.remove(fname)
    logger.add_tabular_output(policy_dir+'/'+track_files[0])
    #logger.log_parameters_lite(policy_dir+'/'+track_files[1])
    logger.add_text_output(policy_dir+'/'+track_files[2])

env = TfEnv(normalize(SingleAgentDiscrCommGridWorldGuidedEnv(n = n_goals,
                                                             desc = map_desc,
                                                             agent_escape = args.escape,
                                                             max_timestep = args.pathlen)))

policy = SingleAgentDiscreteMsgMLPPolicy(
    'SingleAgentNet',
    n_row,
    n_col,
    n_goals,
    env_spec=env.spec,
    conv_layers=[64, 32, 16], # number of conv-layers and the number of kernels
    hidden_layers=[32], # hidden layers after conv-layers
    act_dim = 5, # always 5 in grid domain: 4 directions + stay
    hid_func = tf.nn.elu if args.activation is 'elu' else tf.nn.relu,
    opt_msg = args.opt_msg
)

if args.baseline == 'linear':
    baseline = LinearFeatureBaseline(env_spec=env.spec)
else:
    baseline = TFMLPBaseline(env_spec=env.spec, hidden_layers=(64, 32))

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=args.maxsteps,
    max_path_length=args.pathlen,
    n_itr=args.iter,
    discount=args.gamma,
    step_size=args.maxkl,
)

algo.train()
