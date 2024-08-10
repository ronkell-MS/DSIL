import pomdp_py
import random
import math
import numpy as np
import sys
import copy
import itertools
EPSILON = 1e-9
class DecTigerState(pomdp_py.State):
    def __init__(self, position):
        """
        position (tuple): (x1,y1,x2,y2..xn,yn) positions of the agents on the grid.
        box_locations: tuple of size k. (bx1,by1,bx2,by2..bxk,byk)

        (It is so true that the agent's state doesn't need to involve the map!)

        x axis is horizontal. y axis is vertical.
        """

        if type(position) != tuple:
            position = tuple(position)
        self.position = position


    def __hash__(self):
        return hash(self.position)
    def __eq__(self, other):
        if isinstance(other, DecTigerState):
            return self.position == other.position

        else:
            return False

    def __str__(self):
        """
        i changed to give X as seperator and without '(' tokens
        :return:
        """
        """str = self.__repr__()        
        return str.replace(' ','')"""
        state_str = "B"
        for num in self.position:
            state_str+= str(num) +'X'
        return  state_str

    def __repr__(self):
        return "DecTigerState(%s)" % str(self.position)

class DecTigerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, DecTigerAction):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "DecTigerAction(%s)" % self.name

class DecTigerObservation(pomdp_py.Observation):
    def __init__(self, quality):
        self.quality = quality
    def __hash__(self):
        return hash(self.quality)
    def __eq__(self, other):
        if isinstance(other, DecTigerObservation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other
    def __str__(self):
        return str(self.quality)
    def __repr__(self):
        return "DecTigerObservation(%s)" % str(self.quality)

def checkIfMoveAction(action):
    "action is string like upXidle.."
    action_arr = action.split('X')
    move_actions = ['left','right']
    for i in range(len(action_arr)):
        if action_arr[i] in move_actions:
            return True

    return False


class DecTigerTransitionModel(pomdp_py.TransitionModel):

    """ The model is deterministic """




    def move(self, position, action):
        expected = list(position)
        action_arr = action.split('X')
        if action_arr[0] == 'left':
            expected[0]=0
        elif action_arr[0] == 'right':
            expected[0]=1
        elif action_arr[1] == 'left':
            expected[1] =0
        elif action_arr[1]=='right':
            expected[1] = 1
        return tuple(expected)







    def probability(self, next_state, state, action, normalized=False, **kwargs):
        action_name = action.name
        action_arr = action_name.split('X')
        if action_arr[0] == 'open' or action_arr[1] == 'open':
            if state.position[:-1] == next_state.position[:-1]:
                return 0.5
            else:
                return 0
        elif action_arr[0] == action_arr[1]:
            if state.position[0] == state.position[1]:
                if state.position[:-1] == next_state.position[:-1]:
                    return 0.5
                else:
                    return 0
            else:
                if next_state.position == state.position:
                    return 1
                else:
                    return 0
        else:
            next_temp = self.sample(state,action)
            if next_temp.position == next_state.position:
                return 1
            else:
                return 0




    def sample(self, state, action):
        next_position = tuple(state.position)
        action_name = action.name
        action_arr = action_name.split('X')
        if checkIfMoveAction(action_name):
            next_position = self.move(state.position,action_name)
        elif action_arr[0] == 'listen' or action_arr[1] =='listen':
            next_position = next_position
        elif action_arr[0] == 'open' or action_arr[1] =='open':
            which_init_state = random.random()
            init_state  = 1 if which_init_state>0.5 else 0
            next_temp = list(state.position)
            next_temp[-1] = init_state
            next_position = tuple(next_temp)
        elif action_arr[0] == action_arr[1]:
            if state.position[0] == state.position[1] and state.position[0] != state.position[2]: # change to == if you want no change loc
                which_init_state = random.random()
                init_state = 1 if which_init_state > 0.5 else 0
                next_temp = list(state.position)
                next_temp[-1] = init_state
                next_position = tuple(next_temp)
        return DecTigerState(next_position)



    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)
    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)


        NOTE: for more efficieny remove the states where agent loc is in (0,n-1) or (k-1,n-1) because its should be just final state t,t,t,t """
        all_states = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    all_states.append(DecTigerState((i,j,k)))
        return all_states


class DecTigerObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0):
        self.noise = noise

    def probability(self, observation, next_state, action):
        action_name = action.name
        action_arr = action_name.split('X')
        if observation is None and action_arr[0] != 'listen' and action_arr[1] != 'listen':
            return 1-EPSILON
        if action_arr[0] =='listen':
            if next_state.position[0] == next_state.position[-1]:
                if observation.quality == 'Y':
                    return 1-self.noise
                else:
                    return self.noise
            else:
                if observation.quality == 'Y':
                    return self.noise
                else:
                    return 1-self.noise
        elif action_arr[1] == 'listen':
            if next_state.position[1] == next_state.position[-1]:
                if observation.quality == 'Y':
                    return 1 - self.noise
                else:
                    return self.noise
            else:
                if observation.quality == 'Y':
                    return self.noise
                else:
                    return 1 - self.noise

        else:
            return EPSILON

    def sample(self, next_state, action):
        action_name = action.name
        action_arr = action_name.split('X')
        if action_arr[0] =='listen':
            rnd_num = random.random()
            if next_state.position[0] == next_state.position[-1]:
                if rnd_num<self.noise:
                    return DecTigerObservation('N')
                else:
                    return DecTigerObservation('Y')
            else:
                if rnd_num<self.noise:
                    return DecTigerObservation('Y')
                else:
                    return DecTigerObservation('N')
        elif action_arr[1] =='listen':
            rnd_num = random.random()
            if next_state.position[1] == next_state.position[-1]:
                if rnd_num<self.noise:
                    return DecTigerObservation('N')
                else:
                    return DecTigerObservation('Y')
            else:
                if rnd_num<self.noise:
                    return DecTigerObservation('Y')
                else:
                    return DecTigerObservation('N')
        else:
            return None





    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        # return [DecTigerObservation(s)
        #         for s in {"Y", "N"}]
        return [DecTigerObservation('Y'),DecTigerObservation('N')]


class DecTigerRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state, normalized=False, **kwargs):
        action_name = action.name
        action_arr = action_name.split('X')
        if checkIfMoveAction(action_name):
            return -1
        elif action_arr[0] == 'listen' or action_arr[1] == 'listen':
            return -1
        elif action_arr[0] == 'open':
            if state.position[0] == state.position[-1]:
                return -50
            else:
                return 10
        elif action_arr[1] == 'open':
            if state.position[1] == state.position[-1]:
                return -50
            else:
                return 10
        elif action_arr[0] == action_arr[1] and action_arr[0] =='copen':
            if state.position[0] != state.position[1]:
                return -40
            elif state.position[0] == state.position[-1]:
                return -20
            else:
                return 20




    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError


class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self):
        self.ACTIONS=[]
        self.init_actions()



    def init_actions(self):
        actions_list = ['leftXidle', 'rightXidle', 'listenXidle', 'openXidle',
                        'idleXleft', 'idleXright', 'idleXlisten', 'idleXopen',
                        'copenXcopen']
        for i in range(0,len(actions_list)):
            self.ACTIONS.append(DecTigerAction(actions_list[i]))
    """A simple policy model with uniform prior over a
       small, finite action space"""
    # ACTIONS = {DecTigerAction(s)
    #           for s in {"leftXidle", "rightXidle", "listenXidle","openXidle",
    #                     'idleXleft','idleXright','idleXlisten',"idleXopen",
    #                     'copenXcopen'}}


    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return self.ACTIONS

class DecTigerProblem(pomdp_py.POMDP):
    def __init__(self,init_state,init_belief):
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               DecTigerTransitionModel(),
                               DecTigerObservationModel(),
                               DecTigerRewardModel())
        env = pomdp_py.Environment(init_state,
                                   DecTigerTransitionModel(),
                                   DecTigerRewardModel())

        super().__init__(agent, env, name="DecTigerProblem")


def get_all_states_debug():
    """Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)


    NOTE: for more efficieny remove the states where agent loc is in (0,n-1) or (k-1,n-1) because its should be just final state t,t,t,t """
    all_states = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                all_states.append(DecTigerState((i,j,k)))
    return all_states
if __name__ == '__main__':

    init_loc = (0,1,0)
    init_loc2 = (0,1,1)
    init_state = DecTigerState(init_loc)
    init_state2 = DecTigerState(init_loc2)
    all_states = get_all_states_debug()
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    init_bf[init_state] = 0.5
    init_bf[init_state2] = 0.5
    dt_problem = DecTigerProblem(init_state,pomdp_py.Histogram(init_bf))
    while True:
        user_input = input()
        action = DecTigerAction(user_input)
        reward = dt_problem.env.state_transition(action, execute=True)
        observation = dt_problem.agent.observation_model.sample(dt_problem.env.state, action)
        new_belief = pomdp_py.update_histogram_belief(dt_problem.agent.cur_belief,
                                                      action, observation,
                                                      dt_problem.agent.observation_model,
                                                      dt_problem.agent.transition_model)
        dt_problem.agent.set_belief(new_belief)

    print('hello')