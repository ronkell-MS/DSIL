# This is a sample Python script.
import pomdp_py
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math



from pomdp_problems.boxpushing.box_push_problem import BoxPushginSampleProblem, BoxPushingSampleAction, BoxPushingSampleState, BoxPushingSampleObservation, checkIfCheckAction, checkIfSinglePushAction, checkIfCollabPush, checkIfMoveAction, get_all_states_dubg
import time
from pomdp_py import to_pomdp_file
from pomdp_py import to_pomdpx_file
from pomdp_py import vi_pruning
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import itertools


def my_to_pomdp_file(agent, output_path=None,
                  discount_factor=0.95):
    """
    Pass in an Agent, and use its components to generate
    a .pomdp file to `output_path`.

    The .pomdp file format is specified at:
    http://www.pomdp.org/code/pomdp-file-spec.html

    Note:

    * It is assumed that the reward is independent of the observation.
    * The state, action, and observations of the agent must be
      explicitly enumerable.
    * The state, action and observations of the agent must be
      convertable to a string that does not contain any blank space.

    Args:
        agent (~pomdp_py.framework.basics.Agent): The agent
        output_path (str): The path of the output file to write in. Optional.
                           Default None.
        discount_factor (float): The discount factor
    Returns:
        (list, list, list): The list of states, actions, observations that
           are ordered in the same way as they are in the .pomdp file.
    """
    # Preamble
    try:
        all_states = list(agent.all_states)
        all_actions = list(agent.all_actions)
        all_observations = list(agent.all_observations)
    except NotImplementedError:
        raise ValueError("S, A, O must be enumerable for a given agent to convert to .pomdp format")

    content = "discount: %f\n" % discount_factor
    content += "values: reward\n" # We only consider reward, not cost.

    list_of_states = " ".join(str(s) for s in all_states)
    assert len(list_of_states.split(" ")) == len(all_states),\
        "states must be convertable to strings without blank spaces"
    content += "states: %s\n" % list_of_states

    list_of_actions = " ".join(str(a) for a in all_actions)
    assert len(list_of_actions.split(" ")) == len(all_actions),\
        "actions must be convertable to strings without blank spaces"
    content += "actions: %s\n" % list_of_actions

    list_of_observations = " ".join(str(o) for o in all_observations)
    assert len(list_of_observations.split(" ")) == len(all_observations),\
        "observations must be convertable to strings without blank spaces"
    content += "observations: %s\n" % list_of_observations

    # Starting belief state - they need to be normalized
    total_belief = sum(agent.belief[s] for s in all_states)
    content += "start: %s\n" % (" ".join(["%f" % (agent.belief[s]/total_belief)
                                          for s in all_states]))

    # State transition probabilities - they need to be normalized
    for s in all_states:
        for a in all_actions:
            probs = []
            for s_next in all_states:
                prob = agent.transition_model.probability(s_next, s, a)
                probs.append(prob)
            total_prob = sum(probs)
            if total_prob ==0 :
                print('hello')
            for i, s_next in enumerate(all_states):
                prob_norm = probs[i] / total_prob
                if prob_norm>0:
                    content += 'T : %s : %s : %s %f\n' % (a, s, s_next, prob_norm)

    content += 'O : %s : %s : %s %f\n' % ('*', '*', 'Y', 0.5)
    content += 'O : %s : %s : %s %f\n' % ('*', '*', 'N', 0.5)
    # Observation probabilities - they need to be normalized
    for s_next in all_states:
        for a in all_actions:
            probs = []
            for o in all_observations:
                prob = agent.observation_model.probability(o, s_next, a)
                probs.append(prob)
            total_prob = sum(probs)
            assert total_prob > 0.0,\
                "No observation is probable under state={} action={}"\
                .format(s_next, a)
            for i, o in enumerate(all_observations):
                prob_norm = probs[i] / total_prob
                if prob_norm  >0.8 or prob_norm <0.3:
                    content += 'O : %s : %s : %s %f\n' % (a, s_next, o, prob_norm)

    # Immediate rewards
    for a in all_actions:
        r=0
        if checkIfMoveAction(a.name):
            r=-10
        elif checkIfCheckAction(a.name):
            r = -1
        elif checkIfSinglePushAction(a.name):
            r= -150
        elif checkIfCollabPush(a.name):
            r = -150
        else:
            r = 0
        content += 'R : %s : %s : %s : *  %f\n' % (a, '*', '*', r)


    for s in all_states:
        for a in all_actions:
            for s_next in all_states:
                # We will take the argmax reward, which works for deterministic rewards.
                r = agent.reward_model.sample(s, a, s_next)
                if r == 500 or r == -1000:
                    if s_next == agent.transition_model.sample_deter(s,a):
                        content += 'R : %s : %s : %s : *  %f\n' % (a, s, s_next, r)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(content)
    return all_states, all_actions, all_observations

def create_and_shrink_POMDP(n,k,number_of_agents,agents_init_loc,init_box_locations_lst,box_types,prob_to_push):
    countSmall =0
    countLarge = 0
    for b in box_types:
        if b == 'S':
            countSmall+=1
        else:
            countLarge+=1
    init_state = BoxPushingSampleState(agents_init_loc, init_box_locations_lst[0])
    all_states = get_all_states_dubg(number_of_agents,box_types,n,k)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = []
    for tup in init_box_locations_lst:
        init_state_temp = BoxPushingSampleState(agents_init_loc, tup)
        true_init_bf.append(init_state_temp)
    final_box_loc = [0,n-1]*len(box_types)
    true_init_bf.append(BoxPushingSampleState(('T','T','T','T'), tuple(final_box_loc)))# need tp change this line every time changing target loc
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate]=init_belief_prob

    bp_problem = BoxPushginSampleProblem(n,k,init_state,box_types,number_of_agents,prob_to_push,pomdp_py.Histogram(init_bf))
    filename = f"./boxpushing_pomdp_artifects_refine/myBoxPushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}.pomdp"
    #to_pomdp_file(bp_problem.agent, filename,discount_factor= 0.95)
    my_to_pomdp_file(bp_problem.agent, filename, discount_factor=0.95)

    return

def box_pushing_solve_once(n,k,number_of_agents,agents_init_loc,init_box_locations_lst,box_types,prob_to_push):
    countSmall =0
    countLarge = 0
    for b in box_types:
        if b == 'S':
            countSmall+=1
        else:
            countLarge+=1
    init_state = BoxPushingSampleState(agents_init_loc, init_box_locations_lst[0])
    all_states = get_all_states_dubg(number_of_agents,box_types,n,k)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = []
    for tup in init_box_locations_lst:
        init_state_temp = BoxPushingSampleState(agents_init_loc, tup)
        true_init_bf.append(init_state_temp)

    final_box_loc = [0,n-1]*len(box_types)
    true_init_bf.append(BoxPushingSampleState(('T','T','T','T'), tuple(final_box_loc)))# need tp change this line every time changing target loc
    init_belief_prob = 1 / len(true_init_bf)
    if len(true_init_bf)%3 == 0:
        i= 0
        init_belief_prob_debug = [0.3,0.3,0.4]
        for bstate in true_init_bf:
            init_bf[bstate] = init_belief_prob_debug[i]
            i+=1
    else:
        for bstate in true_init_bf:
            init_bf[bstate]=init_belief_prob


    bp_problem = BoxPushginSampleProblem(n,k,init_state,box_types,number_of_agents,prob_to_push,pomdp_py.Histogram(init_bf))
    # print('init state: ',small_dec_rocksample.env.cur_state)
    pomdpsol_path = "./sarsop/src/pomdpsol"
    policy = sarsop(bp_problem.agent, pomdpsol_path, discount_factor=0.95, timeout=60, memory=1000,
                    precision=0.0001, pomdp_name=f"boxpushing_pomdp_artifects_refine/myBoxPushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}",
                    remove_generated_files=False)
    return policy

def get_str_obs(obs_st):
    if obs_st is None:
        return 'None'
    else:
        return obs_st
def which_agent_operate_now(s):
    arr=s.split('X')
    if arr[0] == 'idle':
        return 2
    elif arr[1] == 'idle':
        return 1
    else:
        return 0
def add_to_trace(bp_problem,action,reward,observation,team_trace,agent1_trace,agent2_trace):
    # --------- adding to traces ----------------#
    lst_state_team =[]
    for i in range(0,len(bp_problem.env.cur_state.position)-1,2):
        lst_state_team.append((bp_problem.env.cur_state.position[i], bp_problem.env.cur_state.position[i+1]))
    for i in range(0,len(bp_problem.env.cur_state.box_locations)-1,2):
        lst_state_team.append((bp_problem.env.cur_state.box_locations[i], bp_problem.env.cur_state.box_locations[i+1]))


    team_trace['states'].append(lst_state_team)

    agent1_trace['states'].append((bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1]))
    agent2_trace['states'].append((bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3]))

    action_name = action.name
    action_arr = action_name.split('X')
    team_trace['actions'].append(action.name)
    agent1_trace['actions'].append(action_arr[0])
    agent2_trace['actions'].append(action_arr[1])

    team_trace['observations'].append(get_str_obs(observation.quality))
    team_trace['rewards'].append(reward)
    agent_act = which_agent_operate_now(action_name)
    if agent_act == 1:
        agent1_trace['observations'].append(get_str_obs(observation.quality))
        agent2_trace['observations'].append('None')

        agent1_trace['rewards'].append(reward)
        agent2_trace['rewards'].append(0)
    elif agent_act ==2:
        agent2_trace['observations'].append(get_str_obs(observation.quality))
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
    action_mirror_ret = BoxPushingSampleAction(action_mirror)

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
def generate_team_traces_for_box_pushing_loadpolicy_refinetrace_fixed_action(n,k,number_of_agents,agents_init_loc,init_box_locations_lst,box_types,prob_to_push):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    countSmall =0
    countLarge = 0
    for b in box_types:
        if b == 'S':
            countSmall+=1
        else:
            countLarge+=1
    init_state = BoxPushingSampleState(agents_init_loc, init_box_locations_lst[0])
    all_states = get_all_states_dubg(number_of_agents,box_types,n,k)
    init_bf = dict()
    for state in all_states:
        init_bf[state] =0
    true_init_bf = []
    for tup in init_box_locations_lst:
        init_state_temp = BoxPushingSampleState(agents_init_loc, tup)
        true_init_bf.append(init_state_temp)
    final_box_loc = [0,n-1]*len(box_types)
    true_init_bf.append(BoxPushingSampleState(('T','T','T','T'), tuple(final_box_loc)))# need tp change this line every time changing target loc
    init_belief_prob = 1 / len(true_init_bf)
    if len(true_init_bf)%3 == 0:
        i= 0
        init_belief_prob_debug = [0.3,0.3,0.4]
        for bstate in true_init_bf:
            init_bf[bstate] = init_belief_prob_debug[i]
            i+=1
    else:
        for bstate in true_init_bf:
            init_bf[bstate]=init_belief_prob

    bp_problem = BoxPushginSampleProblem(n,k,init_state,box_types,number_of_agents,prob_to_push,pomdp_py.Histogram(init_bf))

    policy_path = "%s.policy" % f'boxpushing_pomdp_artifects_refine/myBoxPushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}'
    all_states = list(bp_problem.agent.all_states)
    all_actions = list(bp_problem.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    # if you want to rerun simulate several times

    shared_check_actions = ['check1Xidle','check2Xidle','idleXcheck1','idleXcheck2','check3Xidle','idleXcheck3']


    trace_i = 0

    for i in range(0,3):
        print(i)
        for init_state in true_init_bf:
            print(init_state)
            terminal_state_flag = False
            bp_problem = BoxPushginSampleProblem(n, k, init_state, box_types, number_of_agents, prob_to_push,
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
            skip_injection = 0
            while step<40 and terminal_state_flag==False:
                action = policy.plan(bp_problem.agent)
                reward = bp_problem.env.state_transition(action,execute= True)
                observation = bp_problem.agent.observation_model.sample(bp_problem.env.state,action)
                add_to_trace(bp_problem,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                              action, observation,
                                                              bp_problem.agent.observation_model,
                                                              bp_problem.agent.transition_model)
                curr_belief = bp_problem.agent.cur_belief
                bp_problem.agent.set_belief(new_belief)
                if skip_injection != 1 and observation.quality != None and action.name in shared_check_actions:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        candidates_action_to_inject_agent2 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief, candidates_action_to_inject_agent2,bp_problem)
                    else:
                        candidates_action_to_inject_agent1 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1,bp_problem)
                    if action_to_inject is not None:
                        reward_injected = bp_problem.env.state_transition(action_to_inject, execute=True)
                        observation_injected = bp_problem.agent.observation_model.sample(bp_problem.env.state,
                                                                                    action_to_inject)
                        add_to_trace(bp_problem, action_to_inject, reward_injected, observation_injected, team_trace, agent1_trace,
                                 agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                  action_to_inject, observation_injected,
                                                                  bp_problem.agent.observation_model,
                                                                  bp_problem.agent.transition_model)
                        bp_problem.agent.set_belief(new_belief_after_injection)

                terminal_state_flag = bp_problem.env.cur_state.position[0] =='T'
                step+=1

            if terminal_state_flag == True and len(team_trace['actions'])>= 3:
                team_traces[trace_i] = team_trace
                agent1_traces[trace_i]= agent1_trace
                agent2_traces[trace_i] = agent2_trace
                trace_i += 1
            #print(agent1_trace)
            #print(agent2_trace)
            #print(team_trace['states'])
            #trace_i+=1


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




def traces_for_retrain(traces_to_correct,policy,n,k,number_of_agents,box_types,prob_to_push):
    team_traces = dict()
    agent1_traces=dict()
    agent2_traces= dict()
    countSmall =0
    countLarge = 0
    shared_check_actions = ['check1Xidle','check2Xidle','idleXcheck1','idleXcheck2','check3Xidle','idleXcheck3']
    trace_i = 0
    for trace_ind,trace in traces_to_correct.items():
        for trace_check_point in trace:
            team_trace,agent1_trace,agent2_trace = build_init_traces(trace_check_point['agent1_window'],trace_check_point['agent2_window'],trace_i)
            terminal_state_flag = False
            bp_problem = BoxPushginSampleProblem(n, k, trace_check_point['state'], box_types, number_of_agents, prob_to_push,
                                                 trace_check_point['belief_state'])
            step = 0
            skip_injection = 0
            while step<40 and terminal_state_flag==False:
                action = policy.plan(bp_problem.agent)
                reward = bp_problem.env.state_transition(action,execute= True)
                observation = bp_problem.agent.observation_model.sample(bp_problem.env.state,action)
                add_to_trace(bp_problem,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                              action, observation,
                                                              bp_problem.agent.observation_model,
                                                              bp_problem.agent.transition_model)
                curr_belief = bp_problem.agent.cur_belief
                bp_problem.agent.set_belief(new_belief)
                if skip_injection != 1 and observation.quality != None and action.name in shared_check_actions:
                    action_to_inject = None
                    if which_agent_operate_now(action.name) == 1:
                        candidates_action_to_inject_agent2 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief, new_belief,
                                                                 candidates_action_to_inject_agent2, bp_problem)
                    else:
                        candidates_action_to_inject_agent1 = mirror_action(action.name)
                        action_to_inject = calc_action_to_inject(curr_belief,
                                                                 new_belief,
                                                                 candidates_action_to_inject_agent1, bp_problem)
                    if action_to_inject is not None:
                        reward_injected = bp_problem.env.state_transition(action_to_inject, execute=True)
                        observation_injected = bp_problem.agent.observation_model.sample(bp_problem.env.state,
                                                                                         action_to_inject)
                        add_to_trace(bp_problem, action_to_inject, reward_injected, observation_injected, team_trace,
                                     agent1_trace,
                                     agent2_trace)
                        new_belief_after_injection = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                                      action_to_inject,
                                                                                      observation_injected,
                                                                                      bp_problem.agent.observation_model,
                                                                                      bp_problem.agent.transition_model)
                        bp_problem.agent.set_belief(new_belief_after_injection)

                terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'
                step += 1

            if terminal_state_flag == True and len(team_trace['actions']) >= 3:
                team_traces[trace_i] = team_trace
                agent1_traces[trace_i] = agent1_trace
                agent2_traces[trace_i] = agent2_trace
                trace_i += 1
    return team_traces,agent1_traces,agent2_traces


def genrate_traces_for_retrain(traces_to_correct,policy,agent1_path,agent2_path,team_path,n,k,box_type,number_of_agents,prob_for_push):
    start_time = time.time()
    team, agent1, agent2 = traces_for_retrain(traces_to_correct,policy,n,k,number_of_agents,box_type,prob_for_push)
    end_time = time.time()
    time_to_make_traces = end_time - start_time
    print(f'took {time_to_make_traces} sec to produce traces')
    save_dict(agent1_path,agent1)
    save_dict(agent2_path,agent2)
    save_dict(team_path,team)
    print('done saving traces')
    return team

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



def save_dict(path,dict_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def in_exit_area(n, pos):
    for i in range(0,len(pos),2):
        if pos[i]!=0 or pos[i+1] != n-1:
            return False
    return True


if __name__ == '__main__':
    n = 2
    k = 2
    number_of_agents = 2
    prob_for_push = 0.8
    box_type = ['L','L','L']
    init_agents_location = (1, 0, 1, 0)
    box_init = [(0,n-1),(k-1,0)]
    init_box_locations = list(itertools.product(box_init,repeat = (len(box_type))))
    init_box_location_flatten = []
    for tup in init_box_locations:
        inside_lst=[]
        for inside_tup in tup:
            inside_lst.append(inside_tup[0])
            inside_lst.append(inside_tup[1])
        if not in_exit_area(n,inside_lst):
            init_box_location_flatten.append(tuple(inside_lst))

    countSmall =0
    countLarge = 0
    for b in box_type:
        if b == 'S':
            countSmall+=1
        else:
            countLarge+=1

    schedule = 3
    if schedule ==1: # GENRATE POMDP FILE
        create_and_shrink_POMDP(n,k,number_of_agents,init_agents_location,init_box_location_flatten,box_type,prob_for_push)
    elif schedule == 2:# SOLVE IT USING SARSOP
        policy = box_pushing_solve_once(n,k,number_of_agents,init_agents_location,init_box_location_flatten,box_type,prob_for_push)
    elif schedule ==3:
        start_time = time.time()
        team, agent1, agent2 = generate_team_traces_for_box_pushing_loadpolicy_refinetrace_fixed_action(n,k,number_of_agents,init_agents_location,init_box_location_flatten,box_type,prob_for_push)
        end_time = time.time()
        time_to_make_traces = end_time - start_time
        print(f'took {time_to_make_traces} sec to produce traces')
        save_agents_dicts(f'box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}', '10_08_2023', team, agent1, agent2)
        print('done saving traces')
