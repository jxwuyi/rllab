import numpy as np
from .base import Env
from sandbox.rocky.tf.spaces import Discrete, Box, Product
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import curses

# important!
from cached_property import cached_property

dead_reward = -1
escape_reward = 1
after_escape_reward = 0.05
guided_reward = 0.2
collide_reward = -0.2
hit_wall_reward = -0.2

MAPS = {
    "chain-tiny": [ #MAX 2 agents, 3x5
        ".x.x.",
        ".....",
        "xxxxx"
    ],
    "chain-tiny-fix": [ #MAX 2 agents, 3x5
        "ax.xb",
        "B...A",
        "xxxxx"
    ],
    "chain-mid": [ #MAX 2 agents, 3x7
        ".xx.xx.",
        ".......",
        "xxxxxxx"
    ],
    "chain-mid-fix": [ #MAX 2 agents, 3x7
        "axx.xxb",
        "B.....A",
        "xxxxxxx"
    ],
    "chain": [ #MAX 2 agents, 3x9
        ".xxx.xxx.",
        ".........",
        "xxxxxxxxx"
    ],
    "chain-fix": [ #MAX 2 agents, 3x9
        "axxx.xxxb",
        "B.......A",
        "xxxxxxxxx"
    ],
    "4x4": [
        "....",
        ".xx.",
        ".xx.",
        "...."
    ],
    "4x4-fix": [
        "....",
        "AxxB",
        "bxxa",
        "...."
    ],
    "4x4-single-fix": [
        "....",
        "Axx.",
        ".xxa",
        "...."
    ],
    "5x5": [
        ".....",
        ".xxx.",
        ".....",
        ".xxx.",
        "....."
    ],
    "5x5-fix": [
        "..C..",
        "bxxxa",
        "A...B",
        ".xxx.",
        "..c.."
    ],
    "7x7": [
        "..xxx..",
        ".x...x.",
        "...x...",
        "..xxx..",
        "...x...",
        ".x...x.",
        "..xxx.."
    ],
    "7x7-fix": [
        ".Axxxb.",
        ".x...x.",
        "...x...",
        "dCxxxDc",
        "...x...",
        ".x...x.",
        ".Bxxxa."
    ],
    "9x9": [
        "o...o...o",
        "..x...x..",
        "....o....",
        "..xxxxx..",
        ".........",
        "..xxxxx..",
        "....o....",
        "..x...x..",
        "o...o...o"
    ],
    "9x9-fix": [
        "od..o..co",
        "A.x.E.x.B",
        "....o....",
        "..xxxxx..",
        "d..ba...c",
        "..xxxxx..",
        "....o....",
        "..x.e.x..",
        "o..CoD..o"
    ],
    "9x9-fix-single": [
        "o...o...o",
        "..x.A.x..",
        "....o....",
        "..xxxxx..",
        ".........",
        "..xxxxx..",
        "....o....",
        "..x.a.x..",
        "o...o...o"
    ]
}

def get_num_agent(desc):
    if isinstance(desc, list):
        desc = "".join(desc)
    import string
    last = '.'
    n = 0
    for c in string.ascii_uppercase:
        if c in desc:
            last = c
            n += 1
    if (n > 0 and last != string.ascii_uppercase[n - 1]):
        print ("the input map is not valid! the indices of agents must start from A one by one!")
        assert (False)
    return n

def gen_empty_map_from_desc(desc):
    #generate an empty map
    # >> assume desc = '{n}x{m}-empty'
    import re
    n,m = list(map(int,re.findall('\d+',desc)))
    return [['.'] * m] * n

class SingleAgentDiscrCommGridWorldGuidedEnv(Env, Serializable):
    """
    'A','B',..., 'F' : starting points of agents A, B, C,...,F
    'a', 'b', ..., 'f': goals
    '.': free space
    'x': wall
    'o': hole (terminates episode)
    succeed reward: <escape reward>
    move closer to goal: <guided_reward>

    [NOTE] assume no collision
    """

    # n: number of agents
    def __init__(self, n=2, desc='4x4', seed=0,
                 agent_escape = False,
                 max_timestep = 20):
        np.random.seed(seed)
        assert(n <= 6 and n >= 1);

        Serializable.quick_init(self, locals())
        self.fixed = ('fix' in desc)
        self.input_desc = desc
        self.agent_escape = agent_escape
        self.max_timestep = max_timestep
        assert(isinstance(desc, str))
        if 'empty' in desc:
            desc = gen_empty_map_from_desc(desc)
        else:
            desc = MAPS[desc]
        if self.fixed:
            n_in_map = get_num_agent(desc)
            assert(n_in_map <= 1)
            n = n_in_map

        assert(n > 0) # must be positive number of agents

        desc = np.array(list(map(list, desc)))
        self.raw_desc = desc
        self.raw_desc_backup = desc.copy()
        self.n_row, self.n_col = desc.shape
        self.n_goals = n

        # generate starting locations and goals
        self.gen_start_and_goal()
        self.gen_initial_state()
        self.compute_distance()
        # set original distance
        self.best_dist = self.dist[self.cur_pos]
        self.domain_fig = None
        self.last_action = None
        self.agent_msgs = None
        self.total_timesteps = 0

    # generate start positions and goal positions for agents
    #  --> assume every map is fully connected
    def get_empty_location(self):
        while True:
            x = np.random.randint(0, self.n_row)
            y = np.random.randint(0, self.n_col)
            if self.desc[x,y] == '.':
              return (x, y)
        assert(False)
        return (-1, -1)
    def get_spec_location(self, c):
        for x in range(self.n_row):
            for y in range(self.n_col):
                if self.desc[x, y] == c:
                    return (x, y)
        return (-1, -1)
    def gen_start_and_goal(self):
        """
        Note:
            >> desc: current map
            >> raw_desc: map with goal and intial position
            >> raw_desc_backup: when self.fixed, same as raw_desc;
                                otherwise, the original map information
        """
        self.desc = self.raw_desc_backup.copy()
        self.cur_pos = None
        self.cur_pos_map = [[-1] * self.n_col] * self.n_row
        self.tar_pos = []
        self.is_done = False
        self.obs_id = np.random.randint(self.n_goals)
        # generate goals
        for id in range(self.n_goals):
            # goal position
            c = chr(ord('a') + id)
            x_g, y_g = self.get_spec_location(c)
            if x_g < 0:
                assert (not self.fixed)
                x_g, y_g = self.get_empty_location()
                self.desc[x_g, y_g] = c
            self.tar_pos.append((x_g, y_g))
        # generate agent location
        c = 'A'  # assume single agent
        x_s, y_s = self.get_spec_location(c)
        if x_s < 0:
            assert (not self.fixed)
            x_s, y_s = self.get_empty_location()
            self.desc[x_s, y_s] = c
        self.cur_pos = (x_s, y_s)
        self.cur_pos_map[x_s][y_s] = 0
        # store the generated map if map is not fixed
        if not self.fixed:
            self.raw_desc = self.desc.copy()


    # generate initial states
    #  --> must be called after gen_start_and_goal()
    #  states[0]: global map for obstacles
    #  states[1...n_goals]: agent goals
    #  states[n_goals + 1]: agent location
    #  states[n_goals + 2]: observed goal
    #   > total = n_goals + 3
    def gen_initial_state(self):
        # clear last action
        self.last_action = None
        self.state = np.zeros((self.n_goals + 3, self.n_row, self.n_col))
        # set agent locations
        cp = self.cur_pos
        self.state[self.n_goals + 1][cp[0]][cp[1]] = 1  # current position
        # set goal locations
        for i in range(self.n_goals):
            tp = self.tar_pos[i]
            self.state[1+i][tp[0]][tp[1]] = 1
        # set observed channel
        tp = self.tar_pos[self.obs_id]
        self.state[self.n_goals + 2][tp[0]][tp[1]] = 1
        # set global map
        for x in range(self.n_row):
            for y in range(self.n_col):
                if self.raw_desc[x][y] == 'x':
                    self.state[0][x][y] = 1 # wall
                elif self.raw_desc[x][y] == 'o':
                    self.state[0][x][y] = -1 # hole

    # compute distance to goal for each cell
    #   -1 for can't reach
    def compute_distance(self):
        self.desc = self.raw_desc.copy()
        self.dist = None
        dis = np.ones((self.n_row, self.n_col), dtype=np.int32) * -1
        if self.cur_pos[0] < 0: # already escaped
            self.dist = dis
            return
        gx, gy = self.tar_pos[self.obs_id]
        dis[gx,gy] = 0
        que = [(gx,gy)]
        pt = 0
        while pt < len(que):
            cx, cy = que[pt]
            pt += 1
            for dx, dy in zip([1,-1,0,0],[0,0,-1,1]):
                tx, ty = cx + dx, cy + dy
                if tx < 0 or ty < 0 \
                    or tx >= self.n_row or ty >= self.n_col \
                    or dis[tx, ty] > -1 \
                    or self.desc[tx, ty] in ['#', 'o']:
                    continue
                dis[tx, ty] = dis[cx, cy] + 1
                que.append((tx, ty))
        if dis[self.cur_pos] < 0:
            print(self.desc)
            print(self.cur_pos)
            print(dis)
            print('[ERROR] map not connected!')
            assert(False)
        self.dist = dis

    def reset(self):
        # re-generate all the positions of the agents
        self.gen_start_and_goal()
        # generate observations
        self.gen_initial_state()
        # compute distance
        self.compute_distance()
        # set initial distance
        self.best_dist = self.dist[self.cur_pos]
        #assert self.observation_space.contains(self.state)
        self.total_timesteps = 0
        return self.state

    # printer functions
    def get_current_map(self):
        self.desc = self.raw_desc_backup.copy()
        # clear maps
        x, y = self.get_spec_location('A')
        if x > -1:
            self.desc[x, y] = '.'
        # fill agents
        x,y=self.cur_pos
        if x < 0:
            tx,ty = self.tar_pos[self.obs_id]
            self.desc[tx][ty] = '@'
        else:
            self.desc[x][y] = 'A'
    def get_content_str(self):
        self.get_current_map()
        ret = ''
        for i in range(self.n_row):
            ret += "".join(self.desc[i])+"\n"
        c = 'A'
        a = -1 if self.last_action is None else self.last_action
        ret += c + ' <goal: ' + chr(ord('a')+self.obs_id)+' >'
        ret += " -> " + self.direction_from_action(a) + "\n"
        if self.agent_msgs is not None:
            msg = np.array(self.agent_msgs)
            msg_str = np.array2string(msg,precision=5,separator=',')
            ret += c + ' Msg = ' + msg_str + "\n"
        return ret

    def render(self):
        content = self.get_content_str() + ">>> any key to cont ...\n"
        screen = curses.initscr()
        screen.clear()
        screen.addstr(0, 0, content)
        screen.refresh()
        try:
            screen.getch()
        except KeyboardInterrupt:
            curses.endwin()
            raise KeyboardInterrupt
        curses.endwin()

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3,
            stay=4
        )[d]
    @staticmethod
    def direction_from_action(d):
        if d < 0 or d >= 5:
            return "N/A"
        return ["left","down","right","up","stay"][d]

    # special method for recording msgs
    def add_agent_info(self,info):
        if isinstance(info,dict) and ('msg_0' in info.keys() or 'msg' in info.keys()):
            self.agent_msgs = None
            if 'msg' in info.keys():
                msg = info['msg']
            else:
                msg = info['msg_0']
            self.agent_msgs = msg

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        4: stay
        :param action: an integer in range [0, 4]
        :return:
        """
        # return [(next_state, next_pos, reward, done, prob)]
        possible_next_states = self.get_possible_next_states(action)

        probs = [x[4] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state_info = possible_next_states[next_state_idx]

        reward = next_state_info[2]
        done = next_state_info[3]

        self.total_timesteps += 1
        if self.total_timesteps >= self.max_timestep:
            done = True

        self.cur_pos = next_state_info[1]
        next_state = next_state_info[0]

        self.state = next_state

        self.last_action = action
        return Step(observation=self.state, reward=reward, done=done)

    def get_possible_next_states(self, action):
        """
        Given the action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param action: action, an integer in [0, 4]
        :return: a list of tuples (s', p(s'|s,a))
                 here s' = (observation, next_agent_positions, reward, done)
        """
        # action is a list of int, containing action for each agent

        # if agent_i escapes, pos[i] = [-1, -1]
        coors = self.cur_pos
        assert(coors[0] > -1)  # not escaped

        next_state = self.state.copy()
        reward = 0
        done = False
        increments = [[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]]
        next_coors = None

        # move agents
        x, y = coors
        # single move
        tx, ty = x + increments[action][0], y + increments[action][1]
        # already escape || out of range || hit wall || stay
        if tx < 0 or ty < 0 or tx >= self.n_row or ty >= self.n_col \
            or self.raw_desc[tx][ty] == 'x' \
            or action == 4:  # stay
                next_coors = coors
                if x > -1 and action != 4: # move to wall or out of range
                    reward += hit_wall_reward
        else:
            next_coors = (tx, ty)
            if self.raw_desc[tx][ty] == 'o': # check holes
                reward += dead_reward # dead is terrible
                done = True # game over

        # clear location channels
        next_state[1+self.n_goals][x][y] = 0
        # update next_state and check escape
        x, y = next_coors
        # check escape
        if self.raw_desc[x][y] == chr(ord('a') + self.obs_id): # reach goal
            done = self.agent_escape
            if self.best_dist > 0:
                reward += escape_reward  # a good thing!
                self.best_dist = 0
            else:
                if next_coors == coors:  # stay at the goal
                    reward += after_escape_reward
                else:
                    reward += guided_reward  # already escaped before
        else: # normal move
            # check distance
            if self.dist[next_coors] < self.best_dist:
                self.best_dist = self.dist[next_coors]
            dist_det = self.dist[coors] - self.dist[next_coors]
            if dist_det == 0:  # stay
                reward -= guided_reward / 2  # stay is also bad
            else:
                reward += dist_det * guided_reward

        # fill shared location channels
        next_state[1+self.n_goals][next_coors[0]][next_coors[1]] = 1

        # return, currently assume deterministic transition
        return [(next_state, next_coors, reward, done, 1.)]

    @cached_property
    def action_space(self):
        return Discrete(5)

    @cached_property
    def observation_space(self):
        # channel 0: shared, raw map, -1 ~ hole, 0 ~ free, 1 ~ wall
        # channel 1 + i: agent goals
        # channel n_agent + 1: agent location
        # channel n_agent + 2: observed goal
        return Box(-1.0, 1.0, (self.n_goals + 3, self.n_row,self.n_col))

