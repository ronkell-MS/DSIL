import pomdp_py
import random
import math
import numpy as np
import sys
import copy
import itertools
EPSILON = 1e-9

class BoxPushingSampleState(pomdp_py.State):
    def __init__(self, position, box_locations):
        """
        position (tuple): (x1,y1,x2,y2..xn,yn) positions of the agents on the grid.
        box_locations: tuple of size k. (bx1,by1,bx2,by2..bxk,byk)

        (It is so true that the agent's state doesn't need to involve the map!)

        x axis is horizontal. y axis is vertical.
        """

        if type(position) != tuple:
            position = tuple(position)
        self.position = position
        if type(box_locations) != tuple:
            box_locations = tuple(box_locations)
        self.box_locations = box_locations


    def __hash__(self):
        return hash((self.position, self.box_locations))
    def __eq__(self, other):
        if isinstance(other, BoxPushingSampleState):
            return self.position == other.position\
                and self.box_locations == other.box_locations\

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
        for b in self.box_locations:
            state_str+= str(b) +'X'


        return  state_str

    def __repr__(self):
        return "BoxPushingSampleState(%s|%s)" % (str(self.position), str(self.box_locations))



class BoxPushingSampleAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, BoxPushingSampleAction):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "BoxPushingSampleAction(%s)" % self.name

class BoxPushingSampleObservation(pomdp_py.Observation):
    def __init__(self, quality):
        self.quality = quality
    def __hash__(self):
        return hash(self.quality)
    def __eq__(self, other):
        if isinstance(other, BoxPushingSampleObservation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other
    def __str__(self):
        return str(self.quality)
    def __repr__(self):
        return "BoxPushingSampleObservation(%s)" % str(self.quality)


def checkIfMoveAction(action):
    "action is string like upXidle.."
    action_arr = action.split('X')
    move_actions = ['up','down','left','right']
    for i in range(len(action_arr)):
        if action_arr[i] in move_actions:
            return True

    return False
def checkIfSinglePushAction(action):
    "action is string like sample1Xidle.."
    action_arr = action.split('X')
    single_push_actions =['pushUp','pushDown','pushLeft','pushRight']
    sample_actions =[]
    for i in range(1,4):
        for j in range(len(single_push_actions)):
            action_withboxindex = single_push_actions[j] +str(i)
            sample_actions.append(action_withboxindex)
    for i in range(len(action_arr)):
        if action_arr[i] in sample_actions:
            return True

    return False
def checkIfCollabPush(action):
    "action is string like sample1Xidle.."
    action_arr = action.split('X')
    collab_push_actions =['CpushUp','CpushDown','CpushLeft','CpushRight']
    sample_actions =[]
    action_dict={}
    for i in range(1,4):
        for j in range(len(collab_push_actions)):
            action_withboxindex = collab_push_actions[j] +str(i)
            sample_actions.append(action_withboxindex)

    for i in range(len(action_arr)):
        if action_arr[i] in sample_actions:
            action_dict[action_arr[i]] = action_dict[action_arr[i]] + 1 if action_arr[i] in action_dict.keys() else 1
            if action_dict[action_arr[i]] >= 2:
                return True


    return False

def checkIfCheckAction(action):
    "action is string like check1Xidle.."
    action_arr = action.split('X')
    check_actions = ['check1','check2','check3']
    for i in range(len(action_arr)):
        if action_arr[i] in check_actions:
            return True
    return False
def tupleMoveAction(action):
    "action is string like upXidle.."
    move_single = {
        'up':(0,1),
        'down': (0, -1),
        'left': (-1, 0),
        'right': (1, 0),
        'idle': (0, 0),
    }
    #to compute the permutation want to avoid for less compute
    move_dict_for_threeagents=dict()
    """for key,tup in move_single.items():
        if key =='idle':
            continue
        for i in range(3):
            lst = ['idle','idle','idle']
            lst_loc = [(0,0),(0,0),(0,0)]
            lst[i] =key
            lst_loc[i] =tup
            new_key = 'X'.join(lst)
            new_tup = [element for t in lst_loc for element in t]
            new_tup = tuple(new_tup)
            print(f'{new_key}:{new_tup},')

    return"""

    move_dict={'upXidle':(0,1,0,0),
               'downXidle':(0,-1,0,0),
               'leftXidle':(-1,0,0,0),
               'rightXidle':(1,0,0,0),
               'idleXup':(0,0,0,1),
               'idleXdown':(0,0,0,-1),
               'idleXleft':(0,0,-1,0),
               'idleXright':(0,0,1,0),
               'upXidleXidle':(0,1,0,0,0,0),
               'idleXupXidle':(0,0,0,1,0,0),
               'idleXidleXup':(0,0,0,0,0,1),
               'downXidleXidle': (0, -1, 0, 0, 0, 0),
               'idleXdownXidle': (0, 0, 0, -1, 0, 0),
               'idleXidleXdown': (0, 0, 0, 0, 0, -1),
               'leftXidleXidle': (-1, 0, 0, 0, 0, 0),
               'idleXleftXidle': (0, 0, -1, 0, 0, 0),
               'idleXidleXleft': (0, 0, 0, 0, -1, 0),
               'rightXidleXidle': (1, 0, 0, 0, 0, 0),
               'idleXrightXidle': (0, 0, 1, 0, 0, 0),
               'idleXidleXright': (0, 0, 0, 0, 1, 0),
               }
    return move_dict[action]

class BoxPushingSampleTransitionModel(pomdp_py.TransitionModel):

    """ The model is deterministic """

    def __init__(self, n,k,agent_count,box_types, in_exit_area,prob_for_push):
        """
        box_types = [S,L,S]
        box_locations: array of box locations
        in_exit_area: a function (x1,y1,x2,y2) -> Bool that returns True if boxes in the target grid
        n is numner of rows so its important for y axis
        k is number of cols means its affect the x axis"""
        self.n = n
        self.k = k
        self.agent_count = agent_count
        self._in_exit_area = in_exit_area
        self.box_types = box_types
        self.prob_for_push = prob_for_push


    def _move(self, position, action,n,k):
        move_tup = tupleMoveAction(action)
        expected = []
        for i in range(0,len(position)):
            new_pos = position[i] + move_tup[i]
            if i % 2 ==0: #x axis
                if new_pos < 0 or new_pos >= k:
                    new_pos = position[i]
            else: #y axis
                if new_pos < 0 or new_pos>=n:
                    new_pos = position[i]
            expected.append(new_pos)
        return tuple(expected)
    def singlePush(self,curr_position,action_name,box_pos,box_type,box_locs,box_index,n,k):
        if curr_position != box_pos or box_type != 'S':
            return box_locs

        action_no_num = action_name[:-1]
        action_dict ={'pushUp':(0,1),
                      'pushDown':(0,-1),
                      'pushLeft':(-1,0),
                      'pushRight':(1,0)}
        step = action_dict[action_no_num]
        box_new_pos = (box_pos[0]+step[0],box_pos[1]+step[1])
        if box_new_pos[0] < 0 or box_new_pos[0] >= k or box_new_pos[1]< 0 or box_new_pos[1] >= n:
            return box_locs
        new_boxes_locations = list(box_locs)
        new_boxes_locations[box_index*2] = box_new_pos[0]
        new_boxes_locations[box_index * 2 +1] = box_new_pos[1]
        return tuple(new_boxes_locations)

    def collabPush(self,agents_position,action_arr,box_locs,box_types,n,k):
        agents_acting = []
        for i in range(len(action_arr)):
            if action_arr[i] != 'idle':
                agents_acting.append(i)
        if len(agents_acting) != 2:
            print('BUGGGGG in collabpush')
            return box_locs

        index1 = agents_acting[0]
        index2 = agents_acting[1]
        agent1_position = (agents_position[index1*2],agents_position[index1*2+1])
        agent2_position = (agents_position[index2 * 2], agents_position[index2 * 2 + 1])
        box_index = int(action_arr[index1][-1]) - 1
        box_pos = (box_locs[box_index * 2], box_locs[box_index * 2 + 1])
        box_type = box_types[box_index]

        if agent1_position != agent2_position or agent1_position != box_pos or box_type != 'L':
            return box_locs

        action_no_num = action_arr[index1][:-1]
        action_dict ={'CpushUp':(0,1),
                      'CpushDown':(0,-1),
                      'CpushLeft':(-1,0),
                      'CpushRight':(1,0)}
        step = action_dict[action_no_num]
        box_new_pos = (box_pos[0]+step[0],box_pos[1]+step[1])
        if box_new_pos[0] < 0 or box_new_pos[0] >= k or box_new_pos[1]< 0 or box_new_pos[1] >= n:
            return box_locs
        new_boxes_locations = list(box_locs)
        new_boxes_locations[box_index*2] = box_new_pos[0]
        new_boxes_locations[box_index * 2 +1] = box_new_pos[1]
        return tuple(new_boxes_locations)



    def probability(self, next_state, state, action, normalized=False, **kwargs):
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        deter_state_from_sample = self.sample_deter(state,action)
        if next_state != deter_state_from_sample:
            if state.position == final_state:
                return 0
            elif checkIfMoveAction(action_name) or checkIfCheckAction(action_name):
                return 0
            else:#push scenario
                if state == next_state:
                    return 1-self.prob_for_push
                else:
                    return 0

        else:
            if state.position == final_state:
                return 1
            elif checkIfMoveAction(action_name) or checkIfCheckAction(action_name):
                return 1
            else: #push action
                if state != next_state:
                    return self.prob_for_push
                else:
                    return 1










        #for deterministic use
        """if next_state != self.sample(state, action):
            #return EPSILON
            return 0

        else:
            #return 1.0 - EPSILON
            return 1"""


    def sample(self, state, action):
        next_position = tuple(state.position)
        box_locs = tuple(state.box_locations)
        next_box_locations = box_locs
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        chance_to_execute = random.random()
        if state.position==final_state:
            next_position = final_state
            next_box_locations = box_locs
        else:
            if checkIfMoveAction(action_name):
                next_position = self._move(state.position, action_name,self.n,self.k)
            elif checkIfSinglePushAction(action_name) and chance_to_execute <= self.prob_for_push:
                curr_position = (state.position[0], state.position[1])
                action_index = 0
                for i in range(self.agent_count):
                    if action_arr[i]!= 'idle':
                        curr_position = (state.position[i*2], state.position[i*2+1])
                        action_index = i

                box_index = int(action_arr[action_index][-1]) - 1
                box_pos = (box_locs[box_index*2],box_locs[box_index*2+1])
                box_type = self.box_types[box_index]
                next_box_locations = self.singlePush(curr_position,action_arr[action_index],box_pos,box_type,box_locs,box_index,self.n,self.k)
                if self._in_exit_area(next_box_locations):
                    next_position = final_state
            elif checkIfCollabPush(action_name) and chance_to_execute <= self.prob_for_push:
                next_box_locations = self.collabPush(state.position,action_arr,box_locs,self.box_types,self.n,self.k)
                if self._in_exit_area(next_box_locations):
                    next_position = final_state

        return BoxPushingSampleState(next_position, next_box_locations)
    def sample_deter(self, state, action):
        next_position = tuple(state.position)
        box_locs = tuple(state.box_locations)
        next_box_locations = box_locs
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        if state.position==final_state:
            next_position = final_state
            next_box_locations = box_locs
        else:
            if checkIfMoveAction(action_name):
                next_position = self._move(state.position, action_name,self.n,self.k)
            elif checkIfSinglePushAction(action_name):
                curr_position = (state.position[0], state.position[1])
                action_index = 0
                for i in range(self.agent_count):
                    if action_arr[i]!= 'idle':
                        curr_position = (state.position[i*2], state.position[i*2+1])
                        action_index = i

                box_index = int(action_arr[action_index][-1]) - 1
                box_pos = (box_locs[box_index*2],box_locs[box_index*2+1])
                box_type = self.box_types[box_index]
                next_box_locations = self.singlePush(curr_position,action_arr[action_index],box_pos,box_type,box_locs,box_index,self.n,self.k)
                if self._in_exit_area(next_box_locations):
                    next_position = final_state
            elif checkIfCollabPush(action_name):
                next_box_locations = self.collabPush(state.position,action_arr,box_locs,self.box_types,self.n,self.k)
                if self._in_exit_area(next_box_locations):
                    next_position = final_state

        return BoxPushingSampleState(next_position, next_box_locations)

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)
    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)


        NOTE: for more efficieny remove the states where agent loc is in (0,n-1) or (k-1,n-1) because its should be just final state t,t,t,t """
        all_states = []

        number_of_agents = self.agent_count
        number_of_boxes = len(self.box_types)
        y_axis = list(range(self.n))
        x_axis = list(range(self.k))
        states = list(itertools.product(x_axis,y_axis,repeat = (number_of_agents+number_of_boxes)))
        for i in range(len(states)):
            curr_state_tup = states[i]
            curr_state_lst = list(curr_state_tup)
            only_agents = curr_state_lst[:number_of_agents*2]
            only_boxes = curr_state_lst[number_of_agents*2:]
            if self._in_exit_area(tuple(only_boxes)) == False:
                all_states.append(BoxPushingSampleState(only_agents,only_boxes))
        box_final = [0,self.n-1]*number_of_boxes
        count_temp = number_of_agents * 2
        final_state = tuple(["T"] * count_temp)
        all_states.append(BoxPushingSampleState(final_state,box_final))
        return all_states
class BoxPushingSampleObservationModel(pomdp_py.ObservationModel):
    def __init__(self,agent_count):
        self.agent_count = agent_count

    def probability(self, observation, next_state, action):
        next_box_locs = tuple(next_state.box_locations)
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        if checkIfCheckAction(action_name):
            if next_state.position == final_state:
                if observation.quality is None:
                    return EPSILON
                elif observation.quality == 'N':
                    return 1-EPSILON
                elif observation.quality == 'Y':
                    return EPSILON

            curr_position = (next_state.position[0], next_state.position[1])
            action_index = 0
            for i in range(self.agent_count):
                if action_arr[i] != 'idle':
                    curr_position = (next_state.position[i * 2], next_state.position[i * 2 + 1])
                    action_index = i

            box_index = int(action_arr[action_index][-1]) - 1
            box_pos = (next_box_locs[box_index * 2], next_box_locs[box_index * 2 + 1])

            if observation.quality is None:
                return EPSILON
            elif observation.quality == 'Y' and curr_position==box_pos:
                return 1-EPSILON
            elif observation.quality =='N' and curr_position != box_pos:
                return 1-EPSILON
            else:
                return EPSILON
        else:
            if observation.quality is None:
                return 1 -EPSILON
            else:
                return EPSILON




        """next_box_locs = tuple(next_state.box_locations)
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        if next_state.position== final_state:
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON
        if checkIfCheckAction(action_name):
            curr_position = (next_state.position[0], next_state.position[1])
            action_index = 0
            for i in range(self.agent_count):
                if action_arr[i] != 'idle':
                    curr_position = (next_state.position[i * 2], next_state.position[i * 2 + 1])
                    action_index = i

            box_index = int(action_arr[action_index][-1]) - 1
            box_pos = (next_box_locs[box_index * 2], next_box_locs[box_index * 2 + 1])

            actual_obs = 'None'
            if box_pos == curr_position:
                actual_obs = 'Y'
            else:
                actual_obs = 'N'
            if actual_obs == observation.quality:
                return 1.0 - EPSILON
            else:
                return EPSILON
        else:
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON"""

    def sample(self, next_state, action):
        next_box_locs = tuple(next_state.box_locations)
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        if next_state.position == final_state and checkIfCheckAction(action_name):
            return BoxPushingSampleObservation('N')
        if not next_state.position == final_state and checkIfCheckAction(action_name):
            curr_position = (next_state.position[0], next_state.position[1])
            action_index = 0
            for i in range(self.agent_count):
                if action_arr[i] != 'idle':
                    curr_position = (next_state.position[i * 2], next_state.position[i * 2 + 1])
                    action_index = i

            box_index = int(action_arr[action_index][-1]) - 1
            box_pos = (next_box_locs[box_index * 2], next_box_locs[box_index * 2 + 1])
            if box_pos == curr_position:
                return BoxPushingSampleObservation('Y')
            else:
                return BoxPushingSampleObservation('N')

        else:
            # Terminated or not a check action. So no observation.
            return BoxPushingSampleObservation(None)


    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [BoxPushingSampleObservation(s)
                for s in {"Y", "N"}]

class BoxPushingSampleRewardModel(pomdp_py.RewardModel):
    def __init__(self, agent_count,in_exit_area_for_one_box):
        self.agent_count =agent_count
        self.in_exit_area_for_one_box = in_exit_area_for_one_box

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        box_locs = tuple(state.box_locations)
        next_box_locs = tuple(next_state.box_locations)
        action_name = action.name
        action_arr = action_name.split('X')
        count_temp = self.agent_count * 2
        final_state = tuple(["T"] * count_temp)
        if state.position == final_state:
            return 0
        if checkIfMoveAction(action_name):
            return -10
        if checkIfCheckAction(action_name):
            return -1
        if checkIfSinglePushAction(action_name):
            curr_position = (state.position[0], state.position[1])
            action_index = 0
            for i in range(self.agent_count):
                if action_arr[i] != 'idle':
                    curr_position = (state.position[i * 2], state.position[i * 2 + 1])
                    action_index = i

            box_index = int(action_arr[action_index][-1]) - 1
            box_pos = (box_locs[box_index * 2],box_locs[box_index * 2 + 1])
            next_box_pos = (next_box_locs[box_index * 2], next_box_locs[box_index * 2 + 1])
            if self.in_exit_area_for_one_box(box_pos):
                return -1000
            elif self.in_exit_area_for_one_box(next_box_pos):
                return 500
            else:
                return -30
        if checkIfCollabPush(action_name):
            curr_position = (state.position[0], state.position[1])
            action_index = 0
            for i in range(self.agent_count):
                if action_arr[i] != 'idle':
                    curr_position = (state.position[i * 2], state.position[i * 2 + 1])
                    action_index = i

            box_index = int(action_arr[action_index][-1]) - 1
            box_pos = (box_locs[box_index * 2],box_locs[box_index * 2 + 1])
            next_box_pos = (next_box_locs[box_index * 2], next_box_locs[box_index * 2 + 1])
            if self.in_exit_area_for_one_box(box_pos):
                return -1000
            elif self.in_exit_area_for_one_box(next_box_pos):
                return 500
            else:
                return -20
        return 0



    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, number_of_agents,box_types,n,k):
        self.number_of_agents = number_of_agents
        self.box_types = box_types
        self.n = n
        self.k = k
        self.ACTIONS=[]
        self.init_actions()

    def gen_actions(self):
        actions =[]
        collab_actions =[]
        single_agnet_move =['up','down','left','right']
        single_agent_push = ['pushUp','pushDown','pushLeft','pushRight']
        for i in range(self.number_of_agents):
            for j in range(len(single_agnet_move)):
                temp_lst =['idle'] * self.number_of_agents
                temp_lst[i] = single_agnet_move[j]
                action_str = 'X'.join(temp_lst)
                actions.append(action_str)
        for i in range(self.number_of_agents):
            for j in range(len(self.box_types)):
                temp_lst = ['idle'] * self.number_of_agents
                temp_lst[i] = 'check' + str(j+1)
                action_str = 'X'.join(temp_lst)
                actions.append(action_str)
                for d in range(len(single_agent_push)):
                    action_temp_lst = ['idle'] * self.number_of_agents
                    if self.box_types[j] == 'S':
                        action_temp_lst[i] = single_agent_push[d] + str(j+1)
                        action_push_str = 'X'.join(action_temp_lst)
                        actions.append(action_push_str)
                    else:
                        action_temp_lst[i] = 'C' + single_agent_push[d] + str(j+1)
                        index_next = i+1 if i<self.number_of_agents-1 else 0
                        action_temp_lst[index_next] = 'C' + single_agent_push[d] + str(j+1)
                        action_push_str = 'X'.join(action_temp_lst)
                        if action_push_str not in collab_actions:
                            collab_actions.append(action_push_str)
        actions.extend(collab_actions)
        return actions

        #temp just to test
        """actions = ["upXidle", "downXidle", "leftXidle", "rightXidle", 'idleXup', 'idleXdown', 'idleXleft', 'idleXright',
                   'check1Xidle','idleXcheck1','pushUp1Xidle','pushDown1Xidle','pushLeft1Xidle','pushRight1Xidle','idleXpushUp1','idleXpushDown1','idleXpushLeft1','idleXpushtRight1']
        return actions"""

    def init_actions(self):
        actions = self.gen_actions()
        for i in range(0,len(actions)):
            self.ACTIONS.append(BoxPushingSampleAction(actions[i]))


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

class BoxPushginSampleProblem(pomdp_py.POMDP):
    def in_exit_area(self, pos):
        for i in range(0,len(pos),2):
            if pos[i]!=0 or pos[i+1] != self.n -1:
                return False
        return True
    def in_exit_area_for_one_box(self, pos):
        if pos[0] == 0 and pos[1] == self.n - 1:
            return True
        return False

    def __init__(self, n, k, init_state, box_types,number_of_agents,prob_to_push, init_belief):
        self.n, self.k = n, k
        self.box_types = box_types
        self.number_of_agents = number_of_agents
        self.prob_to_push = prob_to_push
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(number_of_agents,box_types,n,k),
                               BoxPushingSampleTransitionModel(n,k,number_of_agents, box_types, self.in_exit_area,prob_to_push),
                               BoxPushingSampleObservationModel(number_of_agents),
                               BoxPushingSampleRewardModel(number_of_agents,self.in_exit_area_for_one_box))
        env = pomdp_py.Environment(init_state,
                                   BoxPushingSampleTransitionModel(n,k,number_of_agents, box_types, self.in_exit_area,prob_to_push),
                                   BoxPushingSampleRewardModel(number_of_agents,self.in_exit_area_for_one_box))

        super().__init__(agent, env, name="BoxPushingSampleProblem")


# this functions just for debug purpose
def in_exit_area_debug(pos,n):
    for i in range(0,len(pos),2):
        if pos[i]!=0 or pos[i+1] != n -1:
            return False
    return True
def get_all_states_dubg(agent_count,box_types,n,k):
    """Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)


    NOTE: for more efficieny remove the states where agent loc is in (0,n-1) or (k-1,n-1) because its should be just final state t,t,t,t """
    all_states = []

    number_of_agents = agent_count
    number_of_boxes = len(box_types)
    y_axis = list(range(n))
    x_axis = list(range(k))
    states = list(itertools.product(x_axis,y_axis,repeat = (number_of_agents+number_of_boxes)))
    for i in range(len(states)):
        curr_state_tup = states[i]
        curr_state_lst = list(curr_state_tup)
        only_agents = curr_state_lst[:number_of_agents*2]
        only_boxes = curr_state_lst[number_of_agents*2:]
        if in_exit_area_debug(tuple(only_boxes),n) == False:
            all_states.append(BoxPushingSampleState(only_agents,only_boxes))
    box_final = [0,n-1]*number_of_boxes
    count_temp = number_of_agents * 2
    final_state = tuple(["T"] * count_temp)
    all_states.append(BoxPushingSampleState(final_state,box_final))
    return all_states

if __name__ == '__main__':
    init_agents = (1,0,1,1)
    init_box_locations =(1,0)
    init_state = BoxPushingSampleState(init_agents,init_box_locations)
    n=2
    k=2
    number_of_agents=2
    prob_to_push = 0.8
    box_type =['S']
    all_states = get_all_states_dubg(number_of_agents,box_type,n,k)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    init_bf[init_state] = 1
    bp_problem = BoxPushginSampleProblem(n,k,init_state,box_type,number_of_agents,prob_to_push,pomdp_py.Histogram(init_bf))
    while True:
        user_input = input()
        action = BoxPushingSampleAction(user_input)
        reward = bp_problem.env.state_transition(action, execute=True)
        observation = bp_problem.agent.observation_model.sample(bp_problem.env.state, action)
        new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                      action, observation,
                                                      bp_problem.agent.observation_model,
                                                      bp_problem.agent.transition_model)
        bp_problem.agent.set_belief(new_belief)

    print('hello')



