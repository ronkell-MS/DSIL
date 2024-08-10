# This is a sample Python script.
import pomdp_py
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pomdp_problems.tiger.tiger_problem import TigerProblem , TigerState
from pomdp_problems.dec_tiger.dec_tiger_problem import DecTigerProblem, DecTigerState, DecTigerAction
from pomdp_problems.rocksample.rocksample_problem import RockSampleProblem, State, init_particles_belief
from pomdp_problems.dec_rocksample.dec_rocksample_problem import DecRockSampleProblem, DecRockSampleState, DecRockSampleAction
from pomdp_problems.small_dec_rock.small_dec_rock_problem import SmallDecRockSampleProblem, SmallDecRockSampleState, SmallDecRockSampleAction, SmallDecRockSampleObservation, checkIfCheckAction

from pomdp_py import to_pomdp_file
from pomdp_py import to_pomdpx_file
from pomdp_py import vi_pruning
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import itertools
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def get_all_states_for_decRock(_n,specificBound):
    """Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)
    for now doing specific for 3x3 grid with col 1 as collab"""
    all_states = []
    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=3)))
    for x1 in range(0,specificBound+1):
        for y1 in range(0,_n):
            for x2 in range(specificBound,_n):
                for y2 in range(0,_n):
                    for rock in rocks:
                        flag = False
                        if x1 == _n-1 and y1 == 0:
                            flag = True
                        if x2 == _n-1 and y2 == _n-1:
                            flag = True
                        all_states.append(DecRockSampleState(
                            [x1,y1,x2,y2],
                            tuple(rock),
                            flag
                        ))


    return all_states

def small_dec_get_all_states(_n,specificBound):
    """Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)
    for now doing specific for 3x3 grid with col 1 as collab"""
    all_states = []
    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=3)))
    for x1 in range(0,specificBound+1):
        for y1 in range(0,_n):
            for x2 in range(specificBound,_n):
                for y2 in range(0,_n):
                    for rock in rocks:
                        all_states.append(SmallDecRockSampleState(
                            [x1,y1,x2,y2],
                            tuple(rock)
                        ))
    all_states.append(SmallDecRockSampleState(("T","T","T","T"),("G","G","G")))
    return all_states
def which_agent_operate_now(s):
    arr=s.split('X')
    if arr[0] == 'idle':
        return 2
    else:
        return 1
def test_dec_rock():
    init_state = DecRockSampleState((0,0,2,0),('G','G','B'),False)
    rocks_locs = [(0,1),(1,1),(2,1)]
    all_states = get_all_states_for_decRock(3,1)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = [DecRockSampleState((0,0,2,0),('G','G','G'),False),
                                                              DecRockSampleState((0,0,2,0),('G','G','B'),False),
                                                              DecRockSampleState((0,0,2,0),('G','B','G'),False),
                                                              DecRockSampleState((0,0,2,0),('B','G','G'),False),
                                                              DecRockSampleState((0,0,2,0),('B','B','G'),False),
                                                              DecRockSampleState((0,0,2,0),('G','B','B'),False),
                                                              DecRockSampleState((0,0,2,0),('B','G','B'),False),
                                                              DecRockSampleState((0,0,2,0),('B','B','B'),False)]
    for bstate in true_init_bf:
        init_bf[bstate]=0.125

    dec_rocksample = DecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    for step in range(10):
        actionInput = input("enter action")
        action=DecRockSampleAction(actionInput)
        reward = dec_rocksample.env.state_transition(action,execute= True)
        observation = dec_rocksample.agent.observation_model.sample(dec_rocksample.env.state,action)
        bf_print_dict = dict()
        for bf,prob in dec_rocksample.agent.cur_belief.histogram.items():
            if prob>0.0005:
                bf_print_dict[bf] =prob
        print(bf_print_dict)
        print(action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(dec_rocksample.agent.cur_belief,
                                                      action, observation,
                                                      dec_rocksample.agent.observation_model,
                                                      dec_rocksample.agent.transition_model

                                                      )
        dec_rocksample.agent.set_belief(new_belief)

def run_small_dec_rock_sample():
    actions_taken = []
    observation_seen = []
    actual_state = []
    belief_in_step_i=[]
    reward_recieved = []
    terminal_state_flag = False
    init_state = SmallDecRockSampleState((0,0,2,0),('G','G','B'))
    rocks_locs = [(0,1),(1,1),(2,1)]
    all_states = small_dec_get_all_states(3,1)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = [SmallDecRockSampleState((0,0,2,0),('G','G','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','G','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','B','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','G','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','B','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','B','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','G','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','B','B'))]
    for bstate in true_init_bf:
        init_bf[bstate]=0.125

    small_dec_rocksample = SmallDecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    print('init state: ',small_dec_rocksample.env.cur_state)
    #filename = "./dec_final.pomdp"
    #to_pomdp_file(small_dec_rocksample.agent, filename,discount_factor= 0.95)
    #pomdpx_filename = "./temp-pomdp.pomdp"
    #to_pomdpx_file(small_dec_rocksample.agent,pomdpx_filename)
    #return None
    pomdpsol_path = "./sarsop/src/pomdpsol"
    print("start sarsop")
    policy = sarsop(small_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=10000, memory=1000, precision=0.0000000001,pomdp_name= 'dec_final_mini',
                    remove_generated_files=False)
    print('sarsop done solving')
    all_states = list(small_dec_rocksample.agent.all_states)
    all_actions = list(small_dec_rocksample.agent.all_actions)
    #policy_path = "./sarsop/src/shrink_direct.policy"
    #policy = AlphaVectorPolicy.construct(policy_path,
                                         #all_states, all_actions)
    step = 0
    while step<20 and terminal_state_flag==False:
        action = policy.plan(small_dec_rocksample.agent)
        reward = small_dec_rocksample.env.state_transition(action,execute= True)
        observation = small_dec_rocksample.agent.observation_model.sample(small_dec_rocksample.env.state,action)
        bf_print_dict = dict()
        for bf, prob in small_dec_rocksample.agent.cur_belief.histogram.items():
            if prob > 0.0005:
                bf_print_dict[bf] = prob
        print(bf_print_dict)
        print(action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(small_dec_rocksample.agent.cur_belief,
                                                      action, observation,
                                                      small_dec_rocksample.agent.observation_model,
                                                      small_dec_rocksample.agent.transition_model)
        small_dec_rocksample.agent.set_belief(new_belief)
        terminal_state_flag = small_dec_rocksample.env.cur_state.position[0] =='T'
        actions_taken.append(action)
        reward_recieved.append(reward)
        observation_seen.append(observation)
        actual_state.append(small_dec_rocksample.env.cur_state)
        belief_in_step_i.append(bf_print_dict)
        step+=1

    return 1
def get_str_obs(obs_st):
    if obs_st is None:
        return 'None'
    else:
        return obs_st
def generate_team_traces_for_dec_rock():
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    rocks_locs = [(0,1),(1,1),(2,1)]
    all_states = small_dec_get_all_states(3,1)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = [SmallDecRockSampleState((0,0,2,0),('G','G','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','G','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','B','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','G','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','B','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','B','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','G','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','B','B'))]
    for bstate in true_init_bf:
        init_bf[bstate]=0.125

    trace_i=0
    # if you want to rerun simulate several times
    for i in range(0,10):
        for init_state in true_init_bf:
            terminal_state_flag = False
            small_dec_rocksample = SmallDecRockSampleProblem(3,3,init_state,rocks_locs,
                                                  pomdp_py.Histogram(init_bf))
            #print('init state: ',small_dec_rocksample.env.cur_state)
            pomdpsol_path = "./sarsop/src/pomdpsol"
            policy = sarsop(small_dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=10000, memory=1000, precision=0.0000000001,pomdp_name= 'dec_final_mini',
                            remove_generated_files=False)

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
            #--------- done init trace------------------#

            # ---------------- align states and obs for actions so action will be time t and its effect will be in t+1
            #team_trace['states'].append(small_dec_rocksample.env.cur_state)
            #agent1_trace['states'].append((small_dec_rocksample.env.cur_state.position[0], small_dec_rocksample.env.cur_state.position[1]))
            #agent2_trace['states'].append((small_dec_rocksample.env.cur_state.position[2], small_dec_rocksample.env.cur_state.position[3]))
            #agent1_trace['observations'].append('None')
            #agent2_trace['observations'].append('None')
            #team_trace['observations'].append('None')

            step = 0
            while step<20 and terminal_state_flag==False:
                action = policy.plan(small_dec_rocksample.agent)
                reward = small_dec_rocksample.env.state_transition(action,execute= True)
                observation = small_dec_rocksample.agent.observation_model.sample(small_dec_rocksample.env.state,action)
                #--------- adding to traces ----------------#
                team_trace['states'].append(small_dec_rocksample.env.cur_state)
                agent1_trace['states'].append((small_dec_rocksample.env.cur_state.position[0],small_dec_rocksample.env.cur_state.position[1]))
                agent2_trace['states'].append((small_dec_rocksample.env.cur_state.position[2], small_dec_rocksample.env.cur_state.position[3]))

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

                team_trace['beliefs'].append(small_dec_rocksample.agent.cur_belief.histogram)
                #---------done adding to traces------------#
                new_belief = pomdp_py.update_histogram_belief(small_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              small_dec_rocksample.agent.observation_model,
                                                              small_dec_rocksample.agent.transition_model)
                small_dec_rocksample.agent.set_belief(new_belief)
                terminal_state_flag = small_dec_rocksample.env.cur_state.position[0] =='T'
                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            trace_i+=1


    return team_traces,agent1_traces,agent2_traces

def run_small_dec_rock_sample_for_disterbuted_agents():
    actions_taken = []
    observation_seen = []
    actual_state = []
    belief_in_step_i=[]
    reward_recieved = []
    terminal_state_flag = False
    init_state = SmallDecRockSampleState((0,0,2,0),('G','G','B'))
    rocks_locs = [(0,1),(1,1),(2,1)]
    all_states = small_dec_get_all_states(3,1)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = [SmallDecRockSampleState((0,0,2,0),('G','G','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','G','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','B','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','G','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','B','G')),
                                                              SmallDecRockSampleState((0,0,2,0),('G','B','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','G','B')),
                                                              SmallDecRockSampleState((0,0,2,0),('B','B','B'))]
    for bstate in true_init_bf:
        init_bf[bstate]=0.125

    team = SmallDecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    agent1= SmallDecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    agent2= SmallDecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    print('init state: ',team.env.cur_state)
    #filename = "./dec_final.pomdp"
    #to_pomdp_file(small_dec_rocksample.agent, filename,discount_factor= 0.95)
    #pomdpx_filename = "./temp-pomdp.pomdp"
    #to_pomdpx_file(small_dec_rocksample.agent,pomdpx_filename)
    #return None
    pomdpsol_path = "./sarsop/src/pomdpsol"
    print("start sarsop")
    policy = sarsop(team.agent, pomdpsol_path, discount_factor=0.95, timeout=10000, memory=1000, precision=0.0000000001,pomdp_name= 'dec_final_mini',
                    remove_generated_files=False)
    print('sarsop done solving')
    all_states = list(team.agent.all_states)
    all_actions = list(team.agent.all_actions)
    #policy_path = "./sarsop/src/shrink_direct.policy"
    #policy = AlphaVectorPolicy.construct(policy_path,
                                         #all_states, all_actions)
    step = 0
    while step<20 and terminal_state_flag==False:
        action = policy.plan(team.agent)
        agent1_action = policy.plan(agent1.agent)
        agent2_action = policy.plan(agent2.agent)
        reward = team.env.state_transition(action,execute= True)
        observation = team.agent.observation_model.sample(team.env.state,action)


        bf_print_dict = dict()
        for bf, prob in team.agent.cur_belief.histogram.items():
            if prob > 0.0005:
                bf_print_dict[bf] = prob
        print('team_stats')
        print(bf_print_dict)
        print(action, observation, reward)

        bf_agent1_print_dict = dict()
        for bf, prob in agent1.agent.cur_belief.histogram.items():
            if prob > 0.0005:
                bf_agent1_print_dict[bf] = prob
        print()
        print('agent1_stats')
        print(bf_agent1_print_dict)
        print('agent1 action ',agent1_action)
        if which_agent_operate_now(agent1_action.name) == 1 or not (checkIfCheckAction(agent1_action.name)):
            agent1_obs = agent1.agent.observation_model.sample(agent1.env.state, agent1_action)
        else:
            agent1_obs_str = input('enter obs  for agent 1 G/B\n')
            agent1_obs = SmallDecRockSampleObservation(agent1_obs_str)
        print('agent1_obs ',agent1_obs)

        bf_agent2_print_dict = dict()
        for bf, prob in agent2.agent.cur_belief.histogram.items():
            if prob > 0.0005:
                bf_agent2_print_dict[bf] = prob
        print()
        print('agent2_stats')
        print(bf_agent2_print_dict)
        print('agent2 action ',agent2_action)
        if which_agent_operate_now(agent2_action.name) == 2 or not (checkIfCheckAction(agent2_action.name)):
            agent2_obs = agent1.agent.observation_model.sample(agent2.env.state, agent2_action)
        else:
            agent2_obs_str = input('enter obs for agent 2  G/B\n')
            agent2_obs = SmallDecRockSampleObservation(agent2_obs_str)
        print('agent2_obs ', agent2_obs)

        new_belief = pomdp_py.update_histogram_belief(team.agent.cur_belief,
                                                      action, observation,
                                                      team.agent.observation_model,
                                                      team.agent.transition_model)

        new_belief_agent1 = pomdp_py.update_histogram_belief(agent1.agent.cur_belief,
                                                      agent1_action, agent1_obs,
                                                      agent1.agent.observation_model,
                                                      agent1.agent.transition_model)

        new_belief_agent2 = pomdp_py.update_histogram_belief(agent2.agent.cur_belief,
                                                      agent2_action, agent2_obs,
                                                      agent2.agent.observation_model,
                                                      agent2.agent.transition_model)


        team.agent.set_belief(new_belief)
        agent1.agent.set_belief(new_belief_agent1)
        agent2.agent.set_belief(new_belief_agent2)
        terminal_state_flag = team.env.cur_state.position[0] =='T'
        actions_taken.append(action)
        reward_recieved.append(reward)
        observation_seen.append(observation)
        actual_state.append(team.env.cur_state)
        belief_in_step_i.append(bf_print_dict)
        step+=1

    return 1
def run_dec_rock_sample_secVER():
    init_state = DecRockSampleState((0,0,2,0),('G','G','B'),False)
    rocks_locs = [(0,1),(1,1),(2,1)]
    all_states = get_all_states_for_decRock(3,1)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = [DecRockSampleState((0,0,2,0),('G','G','G'),False),
                                                              DecRockSampleState((0,0,2,0),('G','G','B'),False),
                                                              DecRockSampleState((0,0,2,0),('G','B','G'),False),
                                                              DecRockSampleState((0,0,2,0),('B','G','G'),False),
                                                              DecRockSampleState((0,0,2,0),('B','B','G'),False),
                                                              DecRockSampleState((0,0,2,0),('G','B','B'),False),
                                                              DecRockSampleState((0,0,2,0),('B','G','B'),False),
                                                              DecRockSampleState((0,0,2,0),('B','B','B'),False)]
    for bstate in true_init_bf:
        init_bf[bstate]=0.125

    dec_rocksample = DecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram(init_bf))
    filename = "./dec_rock_v2.pomdp"
    #to_pomdp_file(dec_rocksample.agent, filename,discount_factor= 0.95)
    #return None
    pomdpsol_path = "./sarsop/src/pomdpsol"
    print("start sarsop")
    policy = sarsop(dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=30, memory=1000, precision=0.5,pomdp_name= 'dec_rock_v2',
                    remove_generated_files=False)
    print('sarsop done solving')
    for step in range(10):
        action = policy.plan(dec_rocksample.agent)
        reward = dec_rocksample.env.state_transition(action,execute= True)
        observation = dec_rocksample.agent.observation_model.sample(dec_rocksample.env.state,action)
        print(dec_rocksample.agent.cur_belief, action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(dec_rocksample.agent.cur_belief,
                                                      action, observation,
                                                      dec_rocksample.agent.observation_model,
                                                      dec_rocksample.agent.transition_model)
        dec_rocksample.agent.set_belief(new_belief)
    return 1
def run_dec_rock_sample():
    init_state = DecRockSampleState((0,0,2,0),('G','G','B'),False)
    rocks_locs = [(0,1),(1,1),(2,1)]
    dec_rocksample = DecRockSampleProblem(3,3,init_state,rocks_locs,
                                          pomdp_py.Histogram({DecRockSampleState((0,0,2,0),('G','G','G'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('G','G','B'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('G','B','G'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('B','G','G'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('B','B','G'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('G','B','B'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('B','G','B'),False): 0.125,
                                                              DecRockSampleState((0,0,2,0),('B','B','B'),False): 0.125,}))
    filename = "./dec_rock.pomdp"
    #to_pomdp_file(dec_rocksample.agent, filename,discount_factor= 0.95)
    #return None
    pomdpsol_path = "./sarsop/src/pomdpsol"
    print("start sarsop")
    policy = sarsop(dec_rocksample.agent, pomdpsol_path, discount_factor=0.95, timeout=30, memory=1000, precision=0.5,pomdp_name= 'dec_rock',
                    remove_generated_files=False)
    print('sarsop done solving')
    for step in range(10):
        action = policy.plan(dec_rocksample.agent)
        reward = dec_rocksample.env.state_transition(action,execute= True)
        observation = dec_rocksample.agent.observation_model.sample(dec_rocksample.env.state,action)
        print(dec_rocksample.agent.cur_belief, action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(dec_rocksample.agent.cur_belief,
                                                      action, observation,
                                                      dec_rocksample.agent.observation_model,
                                                      dec_rocksample.agent.transition_model)
        dec_rocksample.agent.set_belief(new_belief)
    return 1

def run_dec_tiger_problem():
    init_state = "tiger-left"
    dec_tiger = DecTigerProblem(0.15,DecTigerState(init_state),
                         pomdp_py.Histogram({DecTigerState("tiger-left"): 0.5,
                                             DecTigerState("tiger-right"):0.5}))
    filename = "./test_dec_tiger.pomdp"
    to_pomdp_file(dec_tiger.agent, filename,discount_factor= 0.95)
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(dec_tiger.agent, pomdpsol_path,discount_factor= 0.95, timeout= 10, memory= 20, precision=0.000001,pomdp_name='test_dec_tiger',remove_generated_files=False)
    for step in range(10):
        action = None
        if step == 1 :
            print('here')
            action = DecTigerAction('idleXlisten')
        elif step ==2:
            print('here')
            action = DecTigerAction('listenXidle')
        else:
            action = policy.plan(dec_tiger.agent)
        reward = dec_tiger.env.state_transition(action,execute= True)
        observation = dec_tiger.agent.observation_model.sample(dec_tiger.env.state,action)
        print(dec_tiger.agent.cur_belief, action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(dec_tiger.agent.cur_belief,
                                                      action, observation,
                                                      dec_tiger.agent.observation_model,
                                                      dec_tiger.agent.transition_model)
        dec_tiger.agent.set_belief(new_belief)
    return 1
def run_tiger_problem():
    init_state = "tiger-left"
    tiger = TigerProblem(0.15,TigerState(init_state),
                         pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                             TigerState("tiger-right"):0.5}))
    filename = "./test_tiger.POMDP"
    to_pomdp_file(tiger.agent, filename,discount_factor= 0.95)
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(tiger.agent, pomdpsol_path,discount_factor= 0.95, timeout= 10, memory= 20, precision=0.000001, remove_generated_files=True)
    for step in range(10):
        action = policy.plan(tiger.agent)

        reward = tiger.env.state_transition(action,execute= True)
        observation = tiger.agent.observation_model.sample(tiger.env.state,action)
        print(tiger.agent.cur_belief, action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(tiger.agent.cur_belief,
                                                      action, observation,
                                                      tiger.agent.observation_model,
                                                      tiger.agent.transition_model)
        tiger.agent.set_belief(new_belief)
    return 1

def run_rocksample_problem():
    n, k = 5, 5
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
    belief = "uniform"
    init_belief = init_particles_belief(k, 200, init_state, belief=belief)

    rocksample = RockSampleProblem(n, k, init_state, rock_locs, init_belief)
    filename = "./test_rocksample.POMDP"
    to_pomdp_file(rocksample.agent, filename,discount_factor= 0.95)
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(rocksample.agent, pomdpsol_path,discount_factor= 0.95, timeout= 10, memory= 20, precision=0.000001, remove_generated_files=True)
    for step in range(10):
        action = policy.plan(rocksample.agent)

        reward = rocksample.env.state_transition(action,execute= True)
        observation = rocksample.agent.observation_model.sample(rocksample.env.state,action)
        print(rocksample.agent.cur_belief, action, observation, reward)
        new_belief = pomdp_py.update_histogram_belief(rocksample.agent.cur_belief,
                                                      action, observation,
                                                      rocksample.agent.observation_model,
                                                      rocksample.agent.transition_model)
        rocksample.agent.set_belief(new_belief)
    return 1


def shrink_POMDP(pomdp_path,new_path):
    f = open(pomdp_path,"r")
    f2 = open(new_path,'a')
    count = 0
    first_R = False
    for line in f:
        #line= f.readline()
        arr = line.split(' ')
        if arr[0] == 'R' and first_R == False:
            first_R = True
            example_action = ['R', ':', 'word', ':', '*', ':', '*', ':', '*', '', '-1.000000']
            move_action_set_astrick = {"upXidle", "downXidle", "leftXidle", "rightXidle",
                                       'idleXup', 'idleXdown', 'idleXleft', 'idleXright'}
            check_action_set_astrick = {'check1Xidle', 'check2Xidle',
                                        'idleXcheck2', 'idleXcheck3'}
            sample_action_set = {'sample1Xidle',
                                 'sample2Xidle',
                                 'idleXsample2',
                                 'idleXsample3'}
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
        elif arr[0] == 'R' and float(arr[-1]) <= 0:
            count+=1
        else:
            f2.write(line)

    """example_action = ['R',':','word',':','*',':','*',':','*','','-1.000000']
    move_action_set_astrick= {"upXidle", "downXidle", "leftXidle", "rightXidle",
              'idleXup', 'idleXdown', 'idleXleft', 'idleXright'}
    check_action_set_astrick= {'check1Xidle', 'check2Xidle',
               'idleXcheck2', 'idleXcheck3'}
    sample_action_set = {'sample1Xidle',
              'sample2Xidle',
              'idleXsample2',
              'idleXsample3'}
    for move_action in move_action_set_astrick:
        arr = example_action.copy()
        arr[2] = move_action
        #arr[4] = '*:'
        #arr[6] = '*'
        arr[10] = '-5.000000'
        s= " ".join(arr)
        s+="\n"
        f2.write(s)

    for check_action in check_action_set_astrick:
        arr = example_action.copy()
        arr[2] = check_action
        #arr[4] = '*'
        #arr[6] = '*'
        arr[10] = '-1.000000'
        s= " ".join(arr)
        s+="\n"
        f2.write(s)

    for sample_action in sample_action_set:
        arr = example_action.copy()
        arr[2] = sample_action
        #arr[4] = '*'
        #arr[6] = '*'
        arr[10] = '-10.000000'
        s= " ".join(arr)
        s+="\n"
        f2.write(s)"""

    example_action_terminal = ['R', ':', '*', ':', 'DTXTXTXTXGXGXGX', ':', '*', ':', '*', '', '0.000000']
    s2 = " ".join(example_action_terminal)
    s2 += "\n"
    f2.write(s2)
    f2.close()
    print(count)
    print('done')


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

if __name__ == '__main__':
    print_hi('PyCharm')
    #run_dec_rock_sample()
    #run_dec_tiger_problem()
    #run_tiger_problem()
    #test_dec_rock()
    #run_dec_rock_sample_secVER()

    #original_pomdp_path = "./dec_final.pomdp"
    #new_pomdp_path = "./dec_final_mini.pomdp"
    #shrink_POMDP(original_pomdp_path,new_pomdp_path)
    #run_small_dec_rock_sample()
    #run_small_dec_rock_sample_for_disterbuted_agents()
    team,agent1,agent2 = generate_team_traces_for_dec_rock()
    save_agents_dicts('dec_rock','04_02_2023',team,agent1,agent2)
    #team_dict = load_and_preprocess('traces/dec_rock_28_01_2023_team.pickle')
    #agent1_dict = load_and_preprocess('traces/dec_rock_28_01_2023_agent1.pickle')
    #agent2_dict = load_and_preprocess('traces/dec_rock_28_01_2023_agent2.pickle')
    print('DONE')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
