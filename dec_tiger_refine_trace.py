import pomdp_py
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math




from pomdp_problems.dec_tiger_move.dec_tiger_move_problem import DecTigerProblem, DecTigerAction, DecTigerState, DecTigerObservation, checkIfMoveAction, get_all_states_debug
import time
from pomdp_py import to_pomdp_file
from pomdp_py import to_pomdpx_file
from pomdp_py import vi_pruning
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import itertools


def create_and_shrink_POMDP(init_state,init_bf):
    dt_problem = DecTigerProblem(init_state,pomdp_py.Histogram(init_bf))
    filename = f"./dec_tiger_pomdp_artifects_refine/dec_tiger.pomdp"
    to_pomdp_file(dt_problem.agent, filename,discount_factor= 0.95)

    return

def dec_tiger_solve_once(init_state,init_bf):
    dt_problem = DecTigerProblem(init_state,pomdp_py.Histogram(init_bf))
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(dt_problem.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0000001, pomdp_name=f"dec_tiger_pomdp_artifects_refine/dec_tiger",
                    remove_generated_files=False)
    return policy

def get_str_obs(obs_st):
    if obs_st is None:
        return 'None'
    else:
        return obs_st.quality
def which_agent_operate_now(s):
    arr=s.split('X')
    if arr[0] == 'idle':
        return 2
    elif arr[1] == 'idle':
        return 1
    else:
        return 0
def add_to_trace(dt_problem,action,reward,observation,team_trace,agent1_trace,agent2_trace):
    # --------- adding to traces ----------------#
    team_trace['states'].append(dt_problem.env.cur_state.position)
    agent1_trace['states'].append(dt_problem.env.cur_state.position[0])
    agent2_trace['states'].append(dt_problem.env.cur_state.position[1])

    action_name = action.name
    action_arr = action_name.split('X')
    team_trace['actions'].append(action.name)
    agent1_trace['actions'].append(action_arr[0])
    agent2_trace['actions'].append(action_arr[1])

    team_trace['observations'].append(get_str_obs(observation))
    team_trace['rewards'].append(reward)
    agent_act = which_agent_operate_now(action_name)
    if agent_act == 1:
        agent1_trace['observations'].append(get_str_obs(observation))
        agent2_trace['observations'].append('None')

        agent1_trace['rewards'].append(reward)
        agent2_trace['rewards'].append(0)
    elif agent_act ==2:
        agent2_trace['observations'].append(get_str_obs(observation))
        agent1_trace['observations'].append('None')

        agent1_trace['rewards'].append(0)
        agent2_trace['rewards'].append(reward)
    else:
        agent1_trace['observations'].append('None')
        agent2_trace['observations'].append('None')
        agent1_trace['rewards'].append(reward)
        agent2_trace['rewards'].append(reward)


    #team_trace['beliefs'].append(gen_dec_rocksample.agent.cur_belief.histogram)
    # ---------done adding to traces------------#
def mirror_action(action_name):
    action_arr = action_name.split('X')
    action_mirror ='' +action_arr[1] + 'X' +action_arr[0]
    action_mirror_ret = DecTigerAction(action_mirror)
    return [action_mirror_ret]

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
    if min_kl< 0.2:
        return min_action
    else:
        return None

def generate_team_traces_for_dec_tiger_loadpolicy_refinetrace_fixed_action(init_state, init_bf,true_init_bf,horizon):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    dt_problem = DecTigerProblem(init_state,pomdp_py.Histogram(init_bf))
    policy_path = "%s.policy" % f'dec_tiger_pomdp_artifects_refine/dec_tiger_v2'
    all_states = list(dt_problem.agent.all_states)
    all_actions = list(dt_problem.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times

    shared_check_actions = ['listenXidle','idleXlisten']


    trace_i = 0

    for i in range(0,20):
        print(i)
        for init_state in true_init_bf:
            print(init_state)
            dt_problem = DecTigerProblem(init_state, pomdp_py.Histogram(init_bf))
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
            skip_injection = 0
            while step<horizon:
                curr_state = dt_problem.env.cur_state
                action = policy.plan(dt_problem.agent)
                reward = dt_problem.env.state_transition(action,execute= True)
                observation = dt_problem.agent.observation_model.sample(dt_problem.env.state,action)
                add_to_trace(dt_problem,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(dt_problem.agent.cur_belief,
                                                              action, observation,
                                                              dt_problem.agent.observation_model,
                                                              dt_problem.agent.transition_model)
                curr_belief = dt_problem.agent.cur_belief
                dt_problem.agent.set_belief(new_belief)
                if skip_injection != 1 and observation != None and action.name in shared_check_actions:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        candidates_action_to_inject_agent2 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief, candidates_action_to_inject_agent2,dt_problem)
                    else:
                        candidates_action_to_inject_agent1 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1,dt_problem)
                    if action_to_inject is not None:
                        reward_injected = dt_problem.env.state_transition(action_to_inject, execute=True)
                        observation_injected = dt_problem.agent.observation_model.sample(dt_problem.env.state,
                                                                                    action_to_inject)
                        add_to_trace(dt_problem, action_to_inject, reward_injected, observation_injected, team_trace, agent1_trace,
                                 agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(dt_problem.agent.cur_belief,
                                                                  action_to_inject, observation_injected,
                                                                  dt_problem.agent.observation_model,
                                                                  dt_problem.agent.transition_model)
                        dt_problem.agent.set_belief(new_belief_after_injection)


                step+=1


            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            trace_i += 1
            print(agent1_trace)
            print(agent2_trace)
            print(team_trace['states'])
            #trace_i+=1


    return team_traces,agent1_traces,agent2_traces

def save_dict(path,dict_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def genrate_traces_for_retrain(traces_to_correct,policy,agent1_path,agent2_path,team_path,init_state, init_bf,true_init_bf,horizon):
    start_time = time.time()
    team, agent1, agent2 = traces_for_retrain(traces_to_correct,policy,init_state, init_bf,true_init_bf,horizon)
    end_time = time.time()
    time_to_make_traces = end_time - start_time
    print(f'took {time_to_make_traces} sec to produce traces')
    save_dict(agent1_path,agent1)
    save_dict(agent2_path,agent2)
    save_dict(team_path,team)
    print('done saving traces')
    return team


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





def traces_for_retrain(traces_to_correct,policy,init_state, init_bf,true_init_bf,horizon):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    dt_problem = DecTigerProblem(init_state,pomdp_py.Histogram(true_init_bf))
    policy_path = "%s.policy" % f'dec_tiger_pomdp_artifects_refine/dec_tiger_v2'
    all_states = list(dt_problem.agent.all_states)
    all_actions = list(dt_problem.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times

    shared_check_actions = ['listenXidle','idleXlisten']


    trace_i = 0
    trace_i = 0
    for trace_ind,trace in traces_to_correct.items():
        for trace_check_point in trace:
            team_trace,agent1_trace,agent2_trace = build_init_traces(trace_check_point['agent1_window'],trace_check_point['agent2_window'],trace_i)
            terminal_state_flag = False
            dt_problem = DecTigerProblem(trace_check_point['state'],trace_check_point['belief_state'])
            step = 0             
            skip_injection = 0
            while step<horizon:
                curr_state = dt_problem.env.cur_state
                action = policy.plan(dt_problem.agent)
                reward = dt_problem.env.state_transition(action,execute= True)
                observation = dt_problem.agent.observation_model.sample(dt_problem.env.state,action)
                add_to_trace(dt_problem,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(dt_problem.agent.cur_belief,
                                                              action, observation,
                                                              dt_problem.agent.observation_model,
                                                              dt_problem.agent.transition_model)
                curr_belief = dt_problem.agent.cur_belief
                dt_problem.agent.set_belief(new_belief)
                if skip_injection != 1 and observation != None and action.name in shared_check_actions:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        candidates_action_to_inject_agent2 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief, candidates_action_to_inject_agent2,dt_problem)
                    else:
                        candidates_action_to_inject_agent1 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1,dt_problem)
                    if action_to_inject is not None:
                        reward_injected = dt_problem.env.state_transition(action_to_inject, execute=True)
                        observation_injected = dt_problem.agent.observation_model.sample(dt_problem.env.state,
                                                                                    action_to_inject)
                        add_to_trace(dt_problem, action_to_inject, reward_injected, observation_injected, team_trace, agent1_trace,
                                 agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(dt_problem.agent.cur_belief,
                                                                  action_to_inject, observation_injected,
                                                                  dt_problem.agent.observation_model,
                                                                  dt_problem.agent.transition_model)
                        dt_problem.agent.set_belief(new_belief_after_injection)


                step+=1

            team_traces[trace_i] = team_trace
            agent1_traces[trace_i]= agent1_trace
            agent2_traces[trace_i] = agent2_trace
            trace_i += 1
            print(agent1_trace)
            print(agent2_trace)
            print(team_trace['states'])
            #trace_i+=1

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
    true_init_bf = [init_state,init_state2]
    horizon = 20
    phase = 3

    if phase ==1:#create pomdp
        create_and_shrink_POMDP(init_state,init_bf)
    if phase == 2: # solve pomdp
        policy = dec_tiger_solve_once(init_state,init_bf)
    if phase == 3: #genrate traces
        start_time = time.time()
        team, agent1, agent2 = generate_team_traces_for_dec_tiger_loadpolicy_refinetrace_fixed_action(init_state,init_bf,true_init_bf,horizon)
        end_time = time.time()
        time_to_make_traces = end_time - start_time
        print(f'took {time_to_make_traces} sec to produce traces')
        save_agents_dicts(f'dec_tigerx{horizon}', '26_08_2023', team, agent1, agent2)
        print('done saving traces')