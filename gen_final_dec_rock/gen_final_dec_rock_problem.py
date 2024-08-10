"""
RockSample(n,k) problem
for X rocks
Origin: Heuristic Search Value Iteration for POMDPs (UAI 2004)

Description:

State space:

(x1,y1,x2,y2),(R1,R2,R3..)
Ri = G/B

Action space:

    UP,DOWN,RIGHT,LEFT,check1,check2..
    (up,idle)
    (down,idle)...
    (noise determined by eta (:math:`\eta`). eta=1 -> perfect sensor; eta=0 -> uniform)

Observation: observes the property of rock i when taking Check_i.
G/B

Reward: +10 for Sample a good rock. -10 for Sampling a bad rock.
        move -5
        sense -1
        exit area +10

Initial belief: every rock has equal probability of being Good or Bad.
rock locs is array [(loc1),(loc2),..]
"""
import pomdp_py
import random
import math
import numpy as np
import sys
import copy
import itertools
EPSILON = 1e-9
def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
class GenDecRockSampleState(pomdp_py.State):
    def __init__(self, position, rocktypes):
        """
        position (tuple): (x1,y1,x2,y2) positions of the rovers on the grid.
        rocktypes: tuple of size k. Each is either Good or Bad.
        terminal (bool): The robots are at the terminal state.

        (It is so true that the agent's state doesn't need to involve the map!)

        x axis is horizontal. y axis is vertical.
        """

        if type(position) != tuple:
            position = tuple(position)
        self.position = position
        if type(rocktypes) != tuple:
            rocktypes = tuple(rocktypes)
        self.rocktypes = rocktypes


    def __hash__(self):
        return hash((self.position, self.rocktypes))
    def __eq__(self, other):
        if isinstance(other, GenDecRockSampleState):
            return self.position == other.position\
                and self.rocktypes == other.rocktypes\

        else:
            return False

    def __str__(self):
        """
        i changed to give X as seperator and without '(' tokens
        :return:
        """
        """str = self.__repr__()        
        return str.replace(' ','')"""
        state_str = "D"
        for num in self.position:
            state_str+= str(num) +'X'
        for r in self.rocktypes:
            state_str+= r +'X'


        return  state_str

    def __repr__(self):
        return "GenDecRockSampleState(%s|%s)" % (str(self.position), str(self.rocktypes))

class GenDecRockSampleAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, GenDecRockSampleAction):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "GenDecRockSampleAction(%s)" % self.name

class GenDecRockSampleObservation(pomdp_py.Observation):
    def __init__(self, quality):
        self.quality = quality
    def __hash__(self):
        return hash(self.quality)
    def __eq__(self, other):
        if isinstance(other, GenDecRockSampleObservation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other
    def __str__(self):
        return str(self.quality)
    def __repr__(self):
        return "GenDecRockSampleObservation(%s)" % str(self.quality)

def checkIfMoveAction(action):
    "action is string like upXidle.."
    action_arr = action.split('X')
    move_actions = ['up','down','left','right']
    if action_arr[0] in move_actions or action_arr[1] in move_actions:
        return True
    return False
def checkIfSampleAction(action):
    "action is string like sample1Xidle.."
    action_arr = action.split('X')
    sample_actions = ['sample1','sample2','sample3','sample4','sample5','sample6','sample7','sample8','sample9']
    if action_arr[0] in sample_actions or action_arr[1] in sample_actions:
        return True
    return False

def checkIfCheckAction(action):
    "action is string like check1Xidle.."
    action_arr = action.split('X')
    check_actions = ['check1','check2','check3','check4','check5','check6','check7','check8','check9']
    if action_arr[0] in check_actions or action_arr[1] in check_actions:
        return True
    return False
def tupleMoveAction(action):
    "action is string like upXidle.."
    move_dict={'upXidle':(0,1,0,0),
               'downXidle':(0,-1,0,0),
               'leftXidle':(-1,0,0,0),
               'rightXidle':(1,0,0,0),
               'idleXup':(0,0,0,1),
               'idleXdown':(0,0,0,-1),
               'idleXleft':(0,0,-1,0),
               'idleXright':(0,0,1,0)}
    return move_dict[action]
def invertRockType(rock_type):
    if rock_type == 'G':
        return 'B'
    return 'G'

class GenDecRockSampleTransitionModel(pomdp_py.TransitionModel):

    """ The model is deterministic """

    def __init__(self, n,k,bound, rock_locs, in_exit_area):
        """
        rock_locs: array of rocks locations
        in_exit_area: a function (x1,y1,x2,y2) -> Bool that returns True if (x1,y1,x2,y2) is in exit area
        n is numner of rows so its important for y axis
        k is number of cols means its affect the x axis"""
        self._n = n
        self._k = k
        self._rock_locs = rock_locs
        self._in_exit_area = in_exit_area
        self.specificBound = bound

    def _move_or_exit(self, position, action):
        move_tup = tupleMoveAction(action)
        expected = (position[0] + move_tup[0],
                    position[1] + move_tup[1],
                    position[2] + move_tup[2],
                    position[3] + move_tup[3]
                    )
        if self._in_exit_area(expected):
            return expected, True
        else:
            return (max(0, min(position[0] + move_tup[0], self.specificBound)),
                    max(0, min(position[1] + move_tup[1], self._n-1)),
                    max(self.specificBound, min(position[2] + move_tup[2], self._k - 1)),
                    max(0, min(position[3] + move_tup[3], self._n - 1))
                    ), False

    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            #return EPSILON
            return 0

        else:
            #return 1.0 - EPSILON
            return 1


    def sample(self, state, action):
        next_position = tuple(state.position)
        rocktypes = tuple(state.rocktypes)
        next_rocktypes = rocktypes
        action_name = action.name
        action_arr = action_name.split('X')
        if state.position==("T","T","T","T"):
            next_terminal = True  # already terminated. So no state transition happens
            next_position = ("T","T","T","T")
            next_rocktypes = tuple('G') * len(self._rock_locs)

        else:
            if checkIfMoveAction(action_name):
                next_position, exiting = self._move_or_exit(state.position, action_name)
                if exiting:
                    next_terminal = True
                    next_position = ("T","T","T","T")
                    next_rocktypes = tuple('G') * len(self._rock_locs)
            elif checkIfSampleAction(action_name):
                curr_position = (state.position[0], state.position[1])
                action_index = 0
                if action_arr[0] == 'idle':
                    curr_position = (state.position[2], state.position[3])
                    action_index = 1
                rock_index = int(action_arr[action_index][-1]) - 1
                rock_pos = self._rock_locs[rock_index]
                if rock_pos[0] == curr_position[0] and rock_pos[1] == curr_position[1]:
                    temp_types = list(rocktypes)
                    temp_types[rock_index] ='B'
                    next_rocktypes = tuple(temp_types)

        return GenDecRockSampleState(next_position, next_rocktypes)

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)
    """def get_all_states(self):   REGULAR SETTINGS
        Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        for now doing specific for 3x3 grid with col 1 as collab

        NOTE: for regular settings:more efficieny remove the states where agent loc is in (0,n-1) or (k-1,n-1) because its should be just final state t,t,t,t
         NOTE: for eliran setting : for more efficieny remove the states where agent loc is in (n-1,0) or (k-1,n-1) because its should be just final state t,t,t,t
        all_states = []
        lst = ['G', 'B']
        number_of_rocks = len(self._rock_locs)
        rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
        for x1 in range(0,self.specificBound+1):
            for y1 in range(0,self._n):
                if y1 == self._n -1 and x1 == 0:
                    continue
                for x2 in range(self.specificBound,self._k):
                    for y2 in range(0,self._n):
                        if y2 == self._n-1 and x2 == self._k-1:
                            continue
                        for rock in rocks:
                            all_states.append(GenDecRockSampleState(
                                [x1,y1,x2,y2],
                                tuple(rock)
                            ))
        rock_tup = tuple('G') * len(self._rock_locs)
        all_states.append(GenDecRockSampleState(("T","T","T","T"),rock_tup))
        return all_states"""
    def get_all_states(self):  #eliran settings
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        for now doing specific for 3x3 grid with col 1 as collab

        NOTE: for regular settings:more efficieny remove the states where agent loc is in (0,n-1) or (k-1,n-1) because its should be just final state t,t,t,t
         NOTE: for eliran setting : for more efficieny remove the states where agent loc is in (0,0) or (k-1,n-1) because its should be just final state t,t,t,t"""
        all_states = []
        lst = ['G', 'B']
        number_of_rocks = len(self._rock_locs)
        rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
        for x1 in range(0,self.specificBound+1):
            for y1 in range(0,self._n):
                if y1 == 0 and x1 == 0:
                    continue
                for x2 in range(self.specificBound,self._k):
                    for y2 in range(0,self._n):
                        if y2 == self._n-1 and x2 == self._k-1:
                            continue
                        for rock in rocks:
                            all_states.append(GenDecRockSampleState(
                                [x1,y1,x2,y2],
                                tuple(rock)
                            ))
        rock_tup = tuple('G') * len(self._rock_locs)
        all_states.append(GenDecRockSampleState(("T","T","T","T"),rock_tup))
        return all_states


class GenDecRockSampleObservationModel(pomdp_py.ObservationModel):
    def __init__(self, rock_locs, half_efficiency_dist=20):
        self._half_efficiency_dist = half_efficiency_dist
        self._rock_locs = rock_locs

    def probability(self, observation, next_state, action):
        action_name = action.name
        action_arr = action_name.split('X')
        if next_state.position==("T","T","T","T"):
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON
        if checkIfCheckAction(action_name):
            # compute efficiency
            curr_position = (next_state.position[0], next_state.position[1])
            action_index = 0
            if action_arr[0] == 'idle':
                curr_position = (next_state.position[2],next_state.position[3])
                action_index = 1
            rock_index = int(action_arr[action_index][-1]) -1
            rock_pos = self._rock_locs[rock_index]
            dist = euclidean_dist(rock_pos, curr_position)
            eta = (1 + pow(2, -dist / self._half_efficiency_dist)) * 0.5

            # compute probability
            actual_rocktype = next_state.rocktypes[rock_index]
            if actual_rocktype == observation.quality:
                return eta
            else:
                return 1.0 - eta
        else:
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action):
        action_name = action.name
        action_arr = action_name.split('X')
        if not next_state.position == ("T","T","T","T") and checkIfCheckAction(action_name):
            # compute efficiency
            curr_position = (next_state.position[0], next_state.position[1])
            action_index = 0
            if action_arr[0] == 'idle':
                curr_position = (next_state.position[2],next_state.position[3])
                action_index = 1
            rock_index = int(action_arr[action_index][-1]) -1
            rock_pos = self._rock_locs[rock_index]
            dist = euclidean_dist(rock_pos, curr_position)
            eta = (1 + pow(2, -dist / self._half_efficiency_dist)) * 0.5
            keep = eta > random.uniform(0, 1)

            actual_rocktype = next_state.rocktypes[rock_index]
            if not keep:
                observed_rocktype = invertRockType(actual_rocktype)
                return GenDecRockSampleObservation(observed_rocktype)
            else:
                return GenDecRockSampleObservation(actual_rocktype)
        else:
            # Terminated or not a check action. So no observation.
            return GenDecRockSampleObservation(None)

        return self._probs[next_state][action][observation]


    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [GenDecRockSampleObservation(s)
                for s in {"G", "B"}]


class GenDecRockSampleRewardModel(pomdp_py.RewardModel):
    def __init__(self, rock_locs, in_exit_area):
        self._rock_locs = rock_locs
        self._in_exit_area = in_exit_area

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        if state.position == ("T","T","T","T"):
            return 0  # terminated. No reward
        action_name = action.name
        action_arr = action_name.split('X')
        if checkIfSampleAction(action_name):
            curr_position = (state.position[0], state.position[1])
            action_index = 0
            if action_arr[0] == 'idle':
                curr_position = (state.position[2],state.position[3])
                action_index = 1
            rock_index = int(action_arr[action_index][-1]) -1
            rock_pos = self._rock_locs[rock_index]
            if state.position[0]!= next_state.position[0] or state.position[1]!= next_state.position[1] or state.position[2]!= next_state.position[2] or state.position[3]!= next_state.position[3]:
                return -5
            for i in range(0,len(state.rocktypes)):
                if i != rock_index and state.rocktypes[i] != next_state.rocktypes[i]:
                    return -5

            if rock_pos[0] == curr_position[0] and rock_pos[1] == curr_position[1] and state.rocktypes[rock_index] == 'G' and next_state.rocktypes[rock_index] == 'B':
                return 10
            else:
                return -5
        elif checkIfCheckAction(action_name):
            return -1
        elif checkIfMoveAction(action_name):
            return -2
        return 0





    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError



class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, rock_locs,n,k,bound):
        self._rock_locs = rock_locs
        self._n = n
        self._k = k
        self._bound = bound
        self.ACTIONS=[]
        self.init_actions()

    def gen_actions(self):
        actions = ["upXidle", "downXidle", "leftXidle", "rightXidle", 'idleXup', 'idleXdown', 'idleXleft', 'idleXright']
        for i in range(0, len(self._rock_locs)):
            index=i+1
            temp_loc = self._rock_locs[i]
            if temp_loc[0] < self._bound:
                check_str = 'check' + str(index) + 'Xidle'
                sample_str = 'sample' + str(index) + 'Xidle'
                actions.append(check_str)
                actions.append(sample_str)
            elif temp_loc[0] == self._bound:
                check_str = 'check' + str(index) + 'Xidle'
                sample_str = 'sample' + str(index) + 'Xidle'
                check_str2 = 'idleX' +'check' + str(index)
                sample_str2 = 'idleX' +'sample' + str(index)
                actions.append(check_str)
                actions.append(sample_str)
                actions.append(check_str2)
                actions.append(sample_str2)
            else:
                check_str2 = 'idleX' +'check' + str(index)
                sample_str2 = 'idleX' +'sample' + str(index)
                actions.append(check_str2)
                actions.append(sample_str2)
        return actions

    def init_actions(self):
        actions = self.gen_actions()
        for i in range(0,len(actions)):
            self.ACTIONS.append(GenDecRockSampleAction(actions[i]))


    """A simple policy model with uniform prior over a
       small, finite action space

    actions=["upXidle", "downXidle", "leftXidle", "rightXidle", 'check1Xidle', 'check2Xidle', 'sample1Xidle', 'sample2Xidle',
     'idleXup', 'idleXdown', 'idleXleft', 'idleXright', 'idleXcheck2', 'idleXcheck3', 'idleXsample2', 'idleXsample3']"""

    """ACTIONS=[]
    actions = self.gen_actions()
    for i in range(0,len(actions)):
        ACTIONS.append(GenDecRockSampleAction(actions[i]))"""

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        #return PolicyModel.ACTIONS
        return self.ACTIONS


class GenDecRockSampleProblem(pomdp_py.POMDP):
    """def in_exit_area(self, pos): regular settings
        if pos[0] == 0 and pos[1] == self._n-1:
            return True
        if pos[2] == self._k-1 and pos[3] == self._n-1:
            return True
        return False"""

    def in_exit_area(self, pos): #eliran settings
        if pos[0] == 0 and pos[1] == 0:
            return True
        if pos[2] == self._k-1 and pos[3] == self._n-1:
            return True
        return False

    def __init__(self, n, k,bound, init_state, rock_locs, init_belief):
        self._n, self._k = n, k
        self.specificBound = bound
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(rock_locs,n,k,bound),
                               GenDecRockSampleTransitionModel(n,k,bound, rock_locs, self.in_exit_area),
                               GenDecRockSampleObservationModel(rock_locs),
                               GenDecRockSampleRewardModel(rock_locs, self.in_exit_area))
        env = pomdp_py.Environment(init_state,
                                   GenDecRockSampleTransitionModel(n,k,bound, rock_locs, self.in_exit_area),
                                   GenDecRockSampleRewardModel(rock_locs, self.in_exit_area))
        self._rock_locs = rock_locs
        super().__init__(agent, env, name="GenDecRockSampleProblem")