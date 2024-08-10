"""
1. i will produce POMDP and shrink it not measure time
big note : you need to adapt traces and lstm vocab so there is no end location and instead it will write T ,T for terminal.
2. run sarsop , make traces and save all artifact : record time NOTE : update this phase so we run sarsop onc time and then using it policy mulityple times
3. train lstms record time
4. run simulatin and see success rate.
"""

# This is a sample Python script.
import pomdp_py
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math


from pomdp_problems.gen_final_dec_rock.gen_final_dec_rock_problem import GenDecRockSampleProblem, GenDecRockSampleState, GenDecRockSampleAction, GenDecRockSampleObservation, checkIfCheckAction
import time
from pomdp_py import to_pomdp_file
from pomdp_py import to_pomdpx_file
from pomdp_py import vi_pruning
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import itertools


"""def gen_dec_get_all_states(_n,_k,specificBound,number_of_rocks):   regular settings
    Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)
    for now doing specific for 3x3 grid with col 1 as collab
    all_states = []
    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    for x1 in range(0,specificBound+1):
        for y1 in range(0,_n):
            if y1 == _n - 1 and x1 == 0:
                continue
            for x2 in range(specificBound,_k):
                for y2 in range(0,_n):
                    if y2 == _n - 1 and x2 == _k - 1:
                        continue
                    for rock in rocks:
                        all_states.append(GenDecRockSampleState(
                            [x1,y1,x2,y2],
                            tuple(rock)
                        ))
    rock_tup = tuple('G') * number_of_rocks
    all_states.append(GenDecRockSampleState(("T","T","T","T"),rock_tup))
    return all_states"""

def gen_dec_get_all_states(_n,_k,specificBound,number_of_rocks): #eliran setting
    """Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)
   """
    all_states = []
    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    for x1 in range(0,specificBound+1):
        for y1 in range(0,_n):
            if y1 == 0 and x1 == 0:
                continue
            for x2 in range(specificBound,_k):
                for y2 in range(0,_n):
                    if y2 == _n - 1 and x2 == _k - 1:
                        continue
                    for rock in rocks:
                        all_states.append(GenDecRockSampleState(
                            [x1,y1,x2,y2],
                            tuple(rock)
                        ))
    rock_tup = tuple('G') * number_of_rocks
    all_states.append(GenDecRockSampleState(("T","T","T","T"),rock_tup))
    return all_states


def which_agent_operate_now(s):
    arr=s.split('X')
    if arr[0] == 'idle':
        return 2
    else:
        return 1

def checkIfCheckAction(action):
    "action is string like check1Xidle.."
    action_arr = action.split('X')
    check_actions = ['check1','check2','check3','check4','check5','check6','check7','check8','check9']
    if action_arr[0] in check_actions or action_arr[1] in check_actions:
        return True
    return False


def shrink_POMDP(pomdp_path,new_path,_n,_k,_bound,_rock_locs):

    move_action_set_astrick = {"upXidle", "downXidle", "leftXidle", "rightXidle", 'idleXup', 'idleXdown', 'idleXleft', 'idleXright'}
    check_action_set_astrick = set()
    sample_action_set = set()
    for i in range(0, len(_rock_locs)):
        index= i+1
        temp_loc = _rock_locs[i]
        if temp_loc[0] < _bound:
            check_str = 'check' + str(index) + 'Xidle'
            sample_str = 'sample' + str(index) + 'Xidle'
            check_action_set_astrick.add(check_str)
            sample_action_set.add(sample_str)
        elif temp_loc[0] == _bound:
            check_str = 'check' + str(index) + 'Xidle'
            sample_str = 'sample' + str(index) + 'Xidle'
            check_str2 = 'idleX' + 'check' + str(index)
            sample_str2 = 'idleX' + 'sample' + str(index)
            check_action_set_astrick.add(check_str)
            sample_action_set.add(sample_str)
            check_action_set_astrick.add(check_str2)
            sample_action_set.add(sample_str2)
        else:
            check_str2 = 'idleX' + 'check' + str(index)
            sample_str2 = 'idleX' + 'sample' + str(index)
            check_action_set_astrick.add(check_str2)
            sample_action_set.add(sample_str2)


    f = open(pomdp_path,"r")
    f2 = open(new_path,'a')
    count = 0
    first_R = False
    first_O = False
    for line in f:
        #line= f.readline()
        arr = line.split(' ')
        if arr[0] == 'O' and first_O == False:
            first_O = True
            example_observation = ['O', ':', '*', ':', '*', ':', '*','0.500000']
            o_s = " ".join(example_observation)
            o_s += "\n"
            f2.write(o_s)

        if arr[0] == 'R' and first_R == False:
            first_R = True
            example_action = ['R', ':', 'word', ':', '*', ':', '*', ':', '*', '', '-1.000000']
            for move_action in move_action_set_astrick:
                move_arr = example_action.copy()
                move_arr[2] = move_action
                move_arr[10] = '-2.000000'
                s = " ".join(move_arr)
                s += "\n"
                f2.write(s)

            for check_action in check_action_set_astrick:
                check_arr = example_action.copy()
                check_arr[2] = check_action
                check_arr[10] = '-1.000000'
                s = " ".join(check_arr)
                s += "\n"
                f2.write(s)

            for sample_action in sample_action_set:
                sample_arr = example_action.copy()
                sample_arr[2] = sample_action
                sample_arr[10] = '-5.000000'
                s = " ".join(sample_arr)
                s += "\n"
                f2.write(s)
        if arr[0] == 'T' and float(arr[-1]) == 0:
            count+=1
        elif arr[0] == 'O' and checkIfCheckAction(arr[2]) == False:
            count+=1
        elif arr[0] == 'R' and float(arr[-1]) <= 0:
            count+=1
        else:
            f2.write(line)

    exmp_state = 'DTXTXTXTX'+('GX'*len(_rock_locs))
    example_action_terminal = ['R', ':', '*', ':', exmp_state, ':', '*', ':', '*', '', '0.000000']
    s2 = " ".join(example_action_terminal)
    s2 += "\n"
    f2.write(s2)
    f2.close()
    print(count)
    print('done')

def get_str_obs(obs_st):
    if obs_st is None:
        return 'None'
    else:
        return obs_st

def create_and_shrink_POMDP(n,k,bound,rock_loc_param,agents_init_loc):
    """
    :param n:  y axis
    :param k:  x axis
    :param bound: specific bound is the shared col so agent 1 can move for [0,bound] and agent 2 can mpve from [bound,k-1] in y axis they both can go from [0,n-1]
    :param rock_loc_param: list of tuples of rocks locations
    :param agents_init_loc: tuple of the init loc of the agents
    :return: create POMDP and Shrinked POMDP
    """


    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))


    filename = f"./decRock_pomdp_artifects_refine/DecRockEli{n}x{k}x{number_of_rocks}.pomdp"
    to_pomdp_file(gen_dec_rocksample.agent, filename,discount_factor= 0.95)
    original_pomdp_path = filename
    new_pomdp_path = f"./decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}.pomdp"
    shrink_POMDP(original_pomdp_path,new_pomdp_path,n,k,bound,rock_loc_param)
    return

def dec_rock_solve_once(n,k,bound,rock_loc_param,agents_init_loc):
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(gen_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0001, pomdp_name=f'decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}',
                    remove_generated_files=False)
    return policy
def generate_team_traces_for_dec_rock_solve_once(n,k,bound,rock_loc_param,agents_init_loc):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(gen_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0000000001, pomdp_name=f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}',
                    remove_generated_files=False)
    # if you want to rerun simulate several times
    trace_i = 0
    for i in range(0,10):
        for init_state in true_init_bf:
            terminal_state_flag = False
            gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                                  pomdp_py.Histogram(init_bf))
            team_trace = dict()
            agent1_trace = dict()
            agent2_trace = dict()
            #---------init trace for each compoonent-----#
            team_trace['trace_number'] = trace_i
            agent1_trace['trace_number'] = trace_i
            agent2_trace['trace_number'] = trace_i
            team_trace['actions'] = []
            agent1_trace['actions'] = []
            agent2_trace['actions'] = []
            team_trace['observations'] = []
            agent1_trace['observations'] = []
            agent2_trace['observations'] = []
            team_trace['states'] = []
            agent1_trace['states'] = []
            agent2_trace['states'] = []
            team_trace['rewards'] = []
            agent1_trace['rewards'] = []
            agent2_trace['rewards'] = []
            team_trace['beliefs'] = []

            step = 0
            while step<20 and terminal_state_flag==False:
                action = policy.plan(gen_dec_rocksample.agent)
                reward = gen_dec_rocksample.env.state_transition(action,execute= True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,action)
                #--------- adding to traces ----------------#
                team_trace['states'].append(gen_dec_rocksample.env.cur_state)
                agent1_trace['states'].append((gen_dec_rocksample.env.cur_state.position[0],gen_dec_rocksample.env.cur_state.position[1]))
                agent2_trace['states'].append((gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3]))

                action_name = action.name
                action_arr = action_name.split('X')
                team_trace['actions'].append(action.name)
                agent1_trace['actions'].append(action_arr[0])
                agent2_trace['actions'].append(action_arr[1])

                team_trace['observations'].append(get_str_obs(observation.quality))
                team_trace['rewards'].append(reward)
                if which_agent_operate_now(action_name) == 1:
                    agent1_trace['observations'].append(get_str_obs(observation.quality))
                    agent2_trace['observations'].append('None')

                    agent1_trace['rewards'].append(reward)
                    agent2_trace['rewards'].append(0)
                else:
                    agent2_trace['observations'].append(get_str_obs(observation.quality))
                    agent1_trace['observations'].append('None')

                    agent1_trace['rewards'].append(0)
                    agent2_trace['rewards'].append(reward)

                team_trace['beliefs'].append(gen_dec_rocksample.agent.cur_belief.histogram)
                #---------done adding to traces------------#
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)
                gen_dec_rocksample.agent.set_belief(new_belief)
                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] =='T'
                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            trace_i+=1


    return team_traces,agent1_traces,agent2_traces


def generate_team_traces_for_dec_rock_loadpolicy(n,k,bound,rock_loc_param,agents_init_loc):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    # we will use load to load exist policy.
    """pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(gen_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0000000001, pomdp_name=f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}',
                    remove_generated_files=False)"""
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times
    trace_i = 0
    for i in range(0,1):
        for init_state in true_init_bf:
            terminal_state_flag = False
            gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                                  pomdp_py.Histogram(init_bf))
            team_trace = dict()
            agent1_trace = dict()
            agent2_trace = dict()
            #---------init trace for each compoonent-----#
            team_trace['trace_number'] = trace_i
            agent1_trace['trace_number'] = trace_i
            agent2_trace['trace_number'] = trace_i
            team_trace['actions'] = []
            agent1_trace['actions'] = []
            agent2_trace['actions'] = []
            team_trace['observations'] = []
            agent1_trace['observations'] = []
            agent2_trace['observations'] = []
            team_trace['states'] = []
            agent1_trace['states'] = []
            agent2_trace['states'] = []
            team_trace['rewards'] = []
            agent1_trace['rewards'] = []
            agent2_trace['rewards'] = []
            team_trace['beliefs'] = []

            step = 0
            while step<20 and terminal_state_flag==False:
                action = policy.plan(gen_dec_rocksample.agent)
                reward = gen_dec_rocksample.env.state_transition(action,execute= True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,action)
                #--------- adding to traces ----------------#
                team_trace['states'].append(gen_dec_rocksample.env.cur_state)
                agent1_trace['states'].append((gen_dec_rocksample.env.cur_state.position[0],gen_dec_rocksample.env.cur_state.position[1]))
                agent2_trace['states'].append((gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3]))

                action_name = action.name
                action_arr = action_name.split('X')
                team_trace['actions'].append(action.name)
                agent1_trace['actions'].append(action_arr[0])
                agent2_trace['actions'].append(action_arr[1])

                team_trace['observations'].append(get_str_obs(observation.quality))
                team_trace['rewards'].append(reward)
                if which_agent_operate_now(action_name) == 1:
                    agent1_trace['observations'].append(get_str_obs(observation.quality))
                    agent2_trace['observations'].append('None')

                    agent1_trace['rewards'].append(reward)
                    agent2_trace['rewards'].append(0)
                else:
                    agent2_trace['observations'].append(get_str_obs(observation.quality))
                    agent1_trace['observations'].append('None')

                    agent1_trace['rewards'].append(0)
                    agent2_trace['rewards'].append(reward)

                team_trace['beliefs'].append(gen_dec_rocksample.agent.cur_belief.histogram)
                #---------done adding to traces------------#
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)
                gen_dec_rocksample.agent.set_belief(new_belief)
                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] =='T'
                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            trace_i+=1


    return team_traces,agent1_traces,agent2_traces


def kl_divergence(p_dist, q_dist):
    kl_div = 0.0
    epsilon = 1e-9  # Small value to avoid division by zero or logarithm of zero

    for state in p_dist:
        if state not in q_dist:
            raise ValueError("State '{}' not present in both distributions.".format(state))

        p_prob = p_dist[state]
        q_prob = q_dist[state]

        # Adding epsilon to prevent division by zero and logarithm of zero
        p_prob = max(p_prob, epsilon)
        q_prob = max(q_prob, epsilon)

        kl_div += p_prob * math.log(p_prob / q_prob)

    return kl_div

def calc_action_to_inject(curr_belief, new_belief , candidates, problem_model):
    min_kl =1
    min_action = None
    for action in candidates:
        observation = problem_model.agent.observation_model.sample(problem_model.env.state, action)
        temp_belief = pomdp_py.update_histogram_belief(curr_belief,
                                                      action, observation,
                                                      problem_model.agent.observation_model,
                                                      problem_model.agent.transition_model)
        kl_value = kl_divergence(new_belief.histogram, temp_belief.histogram)
        if kl_value<min_kl :
            min_action = action
            min_kl = kl_value
    if min_kl< 0.5:
        return min_action
    else:
        return None

def add_to_trace(gen_dec_rocksample,action,reward,observation,team_trace,agent1_trace,agent2_trace):
    # --------- adding to traces ----------------#
    team_trace['states'].append(gen_dec_rocksample.env.cur_state)
    agent1_trace['states'].append((gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1]))
    agent2_trace['states'].append((gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3]))

    action_name = action.name
    action_arr = action_name.split('X')
    team_trace['actions'].append(action.name)
    agent1_trace['actions'].append(action_arr[0])
    agent2_trace['actions'].append(action_arr[1])

    team_trace['observations'].append(get_str_obs(observation.quality))
    team_trace['rewards'].append(reward)
    if which_agent_operate_now(action_name) == 1:
        agent1_trace['observations'].append(get_str_obs(observation.quality))
        agent2_trace['observations'].append('None')

        agent1_trace['rewards'].append(reward)
        agent2_trace['rewards'].append(0)
    else:
        agent2_trace['observations'].append(get_str_obs(observation.quality))
        agent1_trace['observations'].append('None')

        agent1_trace['rewards'].append(0)
        agent2_trace['rewards'].append(reward)

    #team_trace['beliefs'].append(gen_dec_rocksample.agent.cur_belief.histogram)
    # ---------done adding to traces------------#
def generate_team_traces_for_dec_rock_loadpolicy_refinetrace(n,k,bound,rock_loc_param,agents_init_loc):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    # we will use load to load exist policy.
    """pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(gen_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0000000001, pomdp_name=f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}',
                    remove_generated_files=False)"""
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times

    candidates_action_to_inject_agent1 = []
    candidates_action_to_inject_agent2 = []
    for action in all_actions:
        if checkIfCheckAction(action.name):
            if which_agent_operate_now(action.name) == 1:
                candidates_action_to_inject_agent1.append(action)
            else:
                candidates_action_to_inject_agent2.append(action)
    trace_i = 0

    for i in range(0,1):
        print(i)
        for init_state in true_init_bf:
            print(init_state)
            terminal_state_flag = False
            gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                                  pomdp_py.Histogram(init_bf))
            team_trace = dict()
            agent1_trace = dict()
            agent2_trace = dict()
            #---------init trace for each compoonent-----#
            team_trace['trace_number'] = trace_i
            agent1_trace['trace_number'] = trace_i
            agent2_trace['trace_number'] = trace_i
            team_trace['actions'] = []
            agent1_trace['actions'] = []
            agent2_trace['actions'] = []
            team_trace['observations'] = []
            agent1_trace['observations'] = []
            agent2_trace['observations'] = []
            team_trace['states'] = []
            agent1_trace['states'] = []
            agent2_trace['states'] = []
            team_trace['rewards'] = []
            agent1_trace['rewards'] = []
            agent2_trace['rewards'] = []
            team_trace['beliefs'] = []

            step = 0
            while step<20 and terminal_state_flag==False:
                action = policy.plan(gen_dec_rocksample.agent)
                reward = gen_dec_rocksample.env.state_transition(action,execute= True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,action)
                add_to_trace(gen_dec_rocksample,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)
                curr_belief = gen_dec_rocksample.agent.cur_belief
                gen_dec_rocksample.agent.set_belief(new_belief)
                if observation.quality != None:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief, candidates_action_to_inject_agent2,gen_dec_rocksample)
                    else:
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1,gen_dec_rocksample)
                    if action_to_inject is not None:
                        reward_injected = gen_dec_rocksample.env.state_transition(action_to_inject, execute=True)
                        observation_injected = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_inject)
                        add_to_trace(gen_dec_rocksample, action_to_inject, reward_injected, observation_injected, team_trace, agent1_trace,
                                 agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_inject, observation_injected,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                        gen_dec_rocksample.agent.set_belief(new_belief_after_injection)

                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] =='T'
                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            trace_i+=1


    return team_traces,agent1_traces,agent2_traces

def mirror_action(action_name):
    action_arr = action_name.split('X')
    action_mirror ='' +action_arr[1] + 'X' +action_arr[0]
    action_mirror_ret = GenDecRockSampleAction(action_mirror)

    return [action_mirror_ret]
def generate_team_traces_for_dec_rock_loadpolicy_refinetrace_fixed_action(n,k,bound,rock_loc_param,agents_init_loc):
    # just inject the mirror action if kl says it is good and keep only check action that are similar for both agents
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    # we will use load to load exist policy.
    """pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(gen_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0000000001, pomdp_name=f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}',
                    remove_generated_files=False)"""
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times

    shared_check_actions = ['check2Xidle','check3Xidle','idleXcheck2','idleXcheck3']


    trace_i = 0

    for i in range(0,1):
        print(i)
        for init_state in true_init_bf:
            print(init_state)
            terminal_state_flag = False
            gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                                  pomdp_py.Histogram(init_bf))
            team_trace = dict()
            agent1_trace = dict()
            agent2_trace = dict()
            #---------init trace for each compoonent-----#
            team_trace['trace_number'] = trace_i
            agent1_trace['trace_number'] = trace_i
            agent2_trace['trace_number'] = trace_i
            team_trace['actions'] = []
            agent1_trace['actions'] = []
            agent2_trace['actions'] = []
            team_trace['observations'] = []
            agent1_trace['observations'] = []
            agent2_trace['observations'] = []
            team_trace['states'] = []
            agent1_trace['states'] = []
            agent2_trace['states'] = []
            team_trace['rewards'] = []
            agent1_trace['rewards'] = []
            agent2_trace['rewards'] = []
            team_trace['beliefs'] = []

            step = 0
            skip_injection = 2
            while step<20 and terminal_state_flag==False:
                action = policy.plan(gen_dec_rocksample.agent)
                reward = gen_dec_rocksample.env.state_transition(action,execute= True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,action)
                add_to_trace(gen_dec_rocksample,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)
                curr_belief = gen_dec_rocksample.agent.cur_belief
                gen_dec_rocksample.agent.set_belief(new_belief)
                if skip_injection != 1 and observation.quality != None and action.name in shared_check_actions:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        candidates_action_to_inject_agent2 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief, candidates_action_to_inject_agent2,gen_dec_rocksample)
                    else:
                        candidates_action_to_inject_agent1 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1,gen_dec_rocksample)
                    if action_to_inject is not None:
                        reward_injected = gen_dec_rocksample.env.state_transition(action_to_inject, execute=True)
                        observation_injected = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_inject)
                        add_to_trace(gen_dec_rocksample, action_to_inject, reward_injected, observation_injected, team_trace, agent1_trace,
                                 agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_inject, observation_injected,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                        gen_dec_rocksample.agent.set_belief(new_belief_after_injection)

                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] =='T'
                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            print(agent1_trace)
            print(agent2_trace)
            trace_i+=1


    return team_traces,agent1_traces,agent2_traces

def build_init_traces(agent1_window,agent2_window,trace_i):
    team_trace = dict()
    agent1_trace = dict()
    agent2_trace = dict()
    # ---------init trace for each compoonent-----#
    team_trace['trace_number'] = trace_i
    agent1_trace['trace_number'] = trace_i
    agent2_trace['trace_number'] = trace_i
    team_trace['actions'] = []
    agent1_trace['actions'] = []
    agent2_trace['actions'] = []
    team_trace['observations'] = []
    agent1_trace['observations'] = []
    agent2_trace['observations'] = []
    team_trace['states'] = []
    agent1_trace['states'] = []
    agent2_trace['states'] = []
    team_trace['rewards'] = []
    agent1_trace['rewards'] = []
    agent2_trace['rewards'] = []
    team_trace['beliefs'] = []

    for i in range(0,len(agent1_window)):
        agent1_trace['actions'].append(agent1_window[i][2])
        agent1_trace['observations'].append(agent1_window[i][3])
        agent1_trace['states'].append(agent1_window[i][1])
    for i in range(0,len(agent2_window)):
        agent2_trace['actions'].append(agent2_window[i][2])
        agent2_trace['observations'].append(agent2_window[i][3])
        agent2_trace['states'].append(agent2_window[i][1])
    j = 0
    while j< max(len(agent1_window),len(agent2_window)):
        action_1 = agent1_window[j][2] if j< len(agent1_window) else 'idle'
        action_2 = agent2_window[j][2] if j < len(agent2_window) else 'idle'
        action_team = action_1+'X'+action_2
        team_trace['actions'].append(action_team)
        j+=1
    return  team_trace,agent1_trace,agent2_trace

def traces_for_retrain(traces_to_correct,n,k,bound,,rock_loc_param,agents_init_loc):
        # just inject the mirror action if kl says it is good and keep only check action that are similar for both agents
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k,bound,number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc,tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    # we will use load to load exist policy.
    """pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(gen_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0000000001, pomdp_name=f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}',
                    remove_generated_files=False)"""
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times

    shared_check_actions = ['check2Xidle','check3Xidle','idleXcheck2','idleXcheck3']

    trace_i = 0
    for trace_ind,trace in traces_to_correct.items():
        for trace_check_point in trace:
            team_trace,agent1_trace,agent2_trace = build_init_traces(trace_check_point['agent1_window'],trace_check_point['agent2_window'],trace_i)
            terminal_state_flag = False
            gen_dec_rocksample = GenDecRockSampleProblem(n,k,bound,trace_check_point['state'],rocks_locs,
                                          trace_check_point['belief_state'])
            
            
            step = 0
            skip_injection = 2
            while step<20 and terminal_state_flag==False:
                action = policy.plan(gen_dec_rocksample.agent)
                reward = gen_dec_rocksample.env.state_transition(action,execute= True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,action)
                add_to_trace(gen_dec_rocksample,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)
                curr_belief = gen_dec_rocksample.agent.cur_belief
                gen_dec_rocksample.agent.set_belief(new_belief)
                if skip_injection != 1 and observation.quality != None and action.name in shared_check_actions:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        candidates_action_to_inject_agent2 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief, candidates_action_to_inject_agent2,gen_dec_rocksample)
                    else:
                        candidates_action_to_inject_agent1 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1,gen_dec_rocksample)
                    if action_to_inject is not None:
                        reward_injected = gen_dec_rocksample.env.state_transition(action_to_inject, execute=True)
                        observation_injected = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_inject)
                        add_to_trace(gen_dec_rocksample, action_to_inject, reward_injected, observation_injected, team_trace, agent1_trace,
                                 agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_inject, observation_injected,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                        gen_dec_rocksample.agent.set_belief(new_belief_after_injection)

                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] =='T'
                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            print(agent1_trace)
            print(agent2_trace)
            trace_i+=1
    return team_traces,agent1_traces,agent2_traces


def load_and_preprocess(pickle_path):
    with open(pickle_path, 'rb') as handle:
        agentdict = pickle.load(handle)
    return agentdict

def save_agents_dicts(problem_name,date,team,agent1,agent2):
    with open(f'traces/{problem_name}_{date}_team.pickle', 'wb') as handle:
        pickle.dump(team, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'traces/{problem_name}_{date}_agent1.pickle', 'wb') as handle:
        pickle.dump(agent1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'traces/{problem_name}_{date}_agent2.pickle', 'wb') as handle:
        pickle.dump(agent2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def scedhule_phases(phase_number,n,k,bound,rocks_locs,agents_init_loc):
    if phase_number==1:
        create_and_shrink_POMDP(n,k,bound,rocks_locs,agents_init_loc)
    elif phase_number == 1.5:
        policy = dec_rock_solve_once(n, k, bound, rocks_locs, agents_init_loc)
    elif phase_number == 2:
        start_time = time.time()
        team, agent1, agent2 = generate_team_traces_for_dec_rock_solve_once(n,k,bound,rocks_locs,agents_init_loc)
        end_time = time.time()
        time_to_make_traces = end_time-start_time
        print(f'took {time_to_make_traces} sec to produce traces')
        save_agents_dicts(f'dec_rock{n}x{k}x{len(rocks_locs)}', '22_07_2023', team, agent1, agent2)
        print('done saving traces')
    elif phase_number ==3:
        start_time = time.time()
        team, agent1, agent2 = generate_team_traces_for_dec_rock_loadpolicy(n, k, bound, rocks_locs, agents_init_loc)
        end_time = time.time()
        time_to_make_traces = end_time - start_time
        print(f'took {time_to_make_traces} sec to produce traces')
        save_agents_dicts(f'dec_rock{n}x{k}x{len(rocks_locs)}', '22_07_2023', team, agent1, agent2)
        print('done saving traces')
    elif phase_number ==4:
        start_time = time.time()
        team, agent1, agent2 = generate_team_traces_for_dec_rock_loadpolicy_refinetrace(n, k, bound, rocks_locs, agents_init_loc)
        end_time = time.time()
        time_to_make_traces = end_time - start_time
        print(f'took {time_to_make_traces} sec to produce traces')
        save_agents_dicts(f'dec_rock{n}x{k}x{len(rocks_locs)}', '05_08_2023', team, agent1, agent2)
        print('done saving traces')
    elif phase_number ==5:
        start_time = time.time()
        team, agent1, agent2 = generate_team_traces_for_dec_rock_loadpolicy_refinetrace_fixed_action(n, k, bound, rocks_locs, agents_init_loc)
        end_time = time.time()
        time_to_make_traces = end_time - start_time
        print(f'took {time_to_make_traces} sec to produce traces')
        save_agents_dicts(f'dec_rock{n}x{k}x{len(rocks_locs)}', '06_08_2023', team, agent1, agent2)
        print('done saving traces')
    else:
        print('done nothing')


if __name__ == '__main__':
#board properties
    print('hello')
    n= 4
    k= 5
    bound = 2
    rocks_locs=[(0,2),(2,3),(2,0),(4,1)]
    agents_init_loc = (0,3,4,0)
    #scedhule_phases(1,n,k,bound,rocks_locs,agents_init_loc) #if need to create pomdp
    #scedhule_phases(1.5,n,k,bound,rocks_locs,agents_init_loc) #if need to solve pomdp first time
    #scedhule_phases(2,n,k,bound,rocks_locs,agents_init_loc)
    #scedhule_phases(3,n,k,bound,rocks_locs,agents_init_loc) # if policy already exist
    #scedhule_phases(4,n,k,bound,rocks_locs,agents_init_loc) # if trace refine
    scedhule_phases(5,n,k,bound,rocks_locs,agents_init_loc) # if trace refine and fixed actions to inject



