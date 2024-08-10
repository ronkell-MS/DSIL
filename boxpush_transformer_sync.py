from pomdp_problems.boxpushing.box_push_problem import BoxPushginSampleProblem, BoxPushingSampleAction, BoxPushingSampleState, BoxPushingSampleObservation, checkIfCheckAction, checkIfSinglePushAction, checkIfCollabPush, checkIfMoveAction, get_all_states_dubg
import time
from pomdp_py import to_pomdp_file
from pomdp_py import to_pomdpx_file
from pomdp_py import vi_pruning
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import itertools
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
# This is a sample Python script.
import pomdp_py
import pickle

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
from boxpushing_refine_trace import genrate_traces_for_retrain

def load_and_preprocess(pickle_path):
    with open(pickle_path, 'rb') as handle:
        agentdict = pickle.load(handle)
    return agentdict
def save_dict(path,dict_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_traces_only_one_idle_dispatch(agent_dict,add_idle_for_collab):
    if add_idle_for_collab:
        return get_traces_only_one_idle_collab_idle(agent_dict)
    else:
        return  get_traces_only_one_idle(agent_dict)


def get_traces_only_one_idle(agent_dict):
  agent_trajectories =[]
  for num,trace_dict in agent_dict.items():
    tr = []
    actions_index = 0
    idle_first = True
    for i in range(len(trace_dict['states'])):
        if trace_dict['actions'][i] !='idle':
            idle_first= True
            tr.append((actions_index,trace_dict['states'][i],trace_dict['actions'][i],trace_dict['observations'][i]))
            actions_index+=1
        elif idle_first ==True:
            idle_first= False
            tr.append((actions_index,trace_dict['states'][i],trace_dict['actions'][i],trace_dict['observations'][i]))
            actions_index+=1
    agent_trajectories.append(tr)
  return agent_trajectories

def get_traces_only_one_idle_collab_idle(agent_dict):
  agent_trajectories =[]
  collab_push_actions = ['CpushUp', 'CpushDown', 'CpushLeft', 'CpushRight']
  for num,trace_dict in agent_dict.items():
    tr = []
    actions_index = 0
    idle_first = True
    for i in range(len(trace_dict['states'])):
        if trace_dict['actions'][i] !='idle':
            if trace_dict['actions'][i][:-1] in collab_push_actions:
                if idle_first == True:
                    ind = max(0,i-1)
                    tr.append((actions_index, trace_dict['states'][ind], 'idle','None'))
                    actions_index+=1
                tr.append((actions_index, trace_dict['states'][i], trace_dict['actions'][i], trace_dict['observations'][i]))
                actions_index += 1
                tr.append((actions_index, trace_dict['states'][i], 'idle', 'None'))
                actions_index+=1
                idle_first = False
            else:
                idle_first= True
                tr.append((actions_index,trace_dict['states'][i],trace_dict['actions'][i],trace_dict['observations'][i]))
                actions_index+=1
        elif idle_first ==True:
            idle_first= False
            tr.append((actions_index,trace_dict['states'][i],trace_dict['actions'][i],trace_dict['observations'][i]))
            actions_index+=1
    if tr[-1][2] == 'idle':
        tr = tr[:-1]
    agent_trajectories.append(tr)
  return agent_trajectories

def save_agents_dicts(problem_name,date,team,agent1,agent2):
    with open(f'traces/{problem_name}_{date}_team.pickle', 'wb') as handle:
        pickle.dump(team, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'traces/{problem_name}_{date}_agent1.pickle', 'wb') as handle:
        pickle.dump(agent1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'traces/{problem_name}_{date}_agent2.pickle', 'wb') as handle:
        pickle.dump(agent2, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_max_len(traces):
  max_len = 0
  for tr in traces:
    max_len=max(max_len,len(tr))
  return max_len

def gen_map_for_locations(n,m):
  loc_map=dict()
  k=0
  for i in range(m):
    for j in range(n):
      loc_map[(i,j)] = k
      k+=1
  loc_map[('T','T')] = k
  return loc_map

def rev_vocab(vocab):
  rev_vocab =dict()
  for key,val in vocab.items():
    rev_vocab[val] = key
  return rev_vocab

class TransformerAgent2(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_actions, max_seq_length):
        super(TransformerAgent2, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, model_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(model_dim, num_actions)
        
    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        encoder_output = self.transformer_encoder(src)
        output = self.fc_out(encoder_output[:, -1, :])  # Use the output of the last token for prediction
        return output


class TransformerAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Reshape to (sequence_length, batch_size, input_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Reshape back to (batch_size, sequence_length, hidden_dim)
        x = self.fc(x[:, -1, :])  # Take the last timestep output
        return x

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories,locs,actions,observations,max_len_traj):
        self.trajectories = trajectories
        self.locs_mapping = locs  # Mapping for actions
        self.action_mapping = actions  # Mapping for actions
        self.observation_mapping = observations  # Mapping for actions
        self.max_len =max_len_traj

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        trajectory = self.trajectories[index]
        input_seq = []
        for t in trajectory[:-1]:
            index,location, action, observation = t
            input_seq.append([index,self.locs_mapping[location], self.action_mapping[action],self.observation_mapping[observation]])
        for i in range(self.max_len - len(input_seq)):
          input_seq.append([-1,-1, -1,-1])
        input_seq = np.array(input_seq, dtype=np.float32)
        target_action = self.action_mapping[trajectory[-1][2]]  # Next action
        return input_seq, target_action
def collate_fn(batch):
    # Sort the batch in descending order based on sequence length
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

    # Pad sequences to match the length of the longest sequence in the batch
    input_seqs, target_actions = zip(*batch)
    padded_input_seqs = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(seq) for seq in input_seqs], batch_first=True, padding_value=-1)
    padded_target_actions = torch.tensor(target_actions, dtype=torch.long)

    return padded_input_seqs, padded_target_actions


def create_all_traj(trajectories):
  ret_traj_list=[]
  for traj in trajectories:
    for i in range(1,len(traj)):
      ret_traj_list.append(traj[0:i+1])
  return ret_traj_list

def predict_next_action(agent, input_seq):
    agent.eval()
    with torch.no_grad():
        output = agent(input_seq.unsqueeze(0))
        #print(input_seq.unsqueeze(0))
        #print(output)
        _, predicted_action = torch.max(output, dim=1)
    agent.train()
    return predicted_action.item()

def embedd_seq(seq,loc_map,action_map,obs_map):
  new_seq = []
  for i in range(len(seq)):
    new_seq.append((seq[i][0],loc_map[seq[i][1]],action_map[seq[i][2]],obs_map[seq[i][3]]))
  return new_seq
def pad_seq(seq,max_len):
  new_seq=seq.copy()
  diff = max_len-len(seq)
  for i in range(diff):
    new_seq.append((-1,-1,-1,-1))
  return new_seq

def save_model(path_to_save,agent_model,optimizer):
    checkpoint = {
        'input_dim': agent_model.input_dim,
        'output_dim': agent_model.output_dim,
        'hidden_dim': agent_model.hidden_dim,
        'num_layers': agent_model.num_layers,
        'num_heads': agent_model.num_heads,
        'state_dict': agent_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path_to_save)


def load_model(path_to_load):
    checkpoint = torch.load(path_to_load)
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    num_heads = checkpoint['num_heads']

    agent = TransformerAgent(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    agent.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adam(agent.parameters(),lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #agent.eval()  # Set the model to evaluation mode
    return agent,optimizer

def train_agent(agent_dict,agent_traces,max_len_agent,agent_model_path,essential_path,n,k,action_map,observation_map):
    start_time = time.time()
    rev_actions = rev_vocab(action_map)
    rev_observations = rev_vocab(observation_map)
    locs = gen_map_for_locations(n, k)
    rev_locs = rev_vocab(locs)
    essentials = {}
    essentials['action_set'] = action_map
    essentials['rev_action_set'] = rev_actions
    essentials['observation_set']  = observation_map
    essentials['rev_observation_set'] = rev_observations
    essentials['locations'] = locs
    essentials['rev_locations'] =rev_locs
    essentials['max_len_in_train'] = max_len_agent
    #end def -----------------------------------------------------------------------

    input_dim = 4  # Dimensionality of the input (index,location, action, observation)
    output_dim = len(action_map.keys())  # Dimensionality of the output (number of actions)
    hidden_dim = 64  # Hidden dimension size
    num_layers = 4  # Number of transformer layers
    num_heads = 8  # Number of attention heads
    batch_size = 64
    learning_rate = 0.001
    epochs = 150
    # Create the Transformer agent
    agent = TransformerAgent(input_dim, output_dim, hidden_dim, num_layers, num_heads)

    # Create the dataset and data loader
    agent_trajectories = agent_traces
    agent_all_trajectories = create_all_traj(agent_trajectories)

    agent_dataset = TrajectoryDataset(agent_all_trajectories, locs, action_map, observation_map,max_len_agent)
    dataloader = DataLoader(agent_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for input_seq, target_action in dataloader:
            #print(input_seq)
            #print(target_action)
            optimizer.zero_grad()

            output = agent(input_seq)
            loss = criterion(output, target_action)
            # print('output is', output)
            # print('target is ', target_action)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    "running example sane check"
    example_1 = agent1_traces[0]
    print(example_1)
    for i in range(2,5):
        example_tr_1 = example_1[:i]
        print(example_tr_1)
        emb_seq = embedd_seq(example_tr_1, locs, action_map, observation_map)
        print(emb_seq)
        padded_seq = pad_seq(emb_seq, max_len_agent1)
        print(padded_seq)
        input_sequence = torch.tensor(padded_seq, dtype=torch.float32)
        predicted_action = predict_next_action(agent, input_sequence)
        print(f"Predicted action: {predicted_action} which is {rev_actions[predicted_action]}")
    example_1 = agent1_traces[1]
    print(example_1)
    for i in range(2,5):
        example_tr_1 = example_1[:i]
        print(example_tr_1)
        emb_seq = embedd_seq(example_tr_1, locs, action_map, observation_map)
        print(emb_seq)
        padded_seq = pad_seq(emb_seq, max_len_agent1)
        print(padded_seq)
        input_sequence = torch.tensor(padded_seq, dtype=torch.float32)
        predicted_action = predict_next_action(agent, input_sequence)
        print(f"Predicted action: {predicted_action} which is {rev_actions[predicted_action]}")

    end_time = time.time()
    time_to_train_agent= end_time - start_time
    print(f'took {time_to_train_agent} sec to train agent')
    save_model(agent_model_path, agent,optimizer)
    save_dict(essential_path,essentials)
    return agent

def retrain_agent(agent_traces,agent_model_path,agent_essential_path,agent_retrained_path):
    agent_dict = load_and_preprocess(agent_essential_path)
    action_map = agent_dict['action_set']
    observation_map = agent_dict['observation_set']
    locs = agent_dict['locations']
    max_len_agent = agent_dict['max_len_in_train']
    start_time = time.time()
    batch_size = 32
    learning_rate = 0.001
    epochs = 5
    # Create the Transformer agent
    agent,optimizer = load_model(agent_model_path)

    # Create the dataset and data loader
    agent_trajectories = agent_traces
    agent_all_trajectories = create_all_traj(agent_trajectories)
    agent_dataset = TrajectoryDataset(agent_all_trajectories, locs, action_map, observation_map,max_len_agent)
    dataloader = DataLoader(agent_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer.param_groups[0]['lr'] = learning_rate
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for input_seq, target_action in dataloader:
            #print(input_seq)
            #print(target_action)
            optimizer.zero_grad()

            output = agent(input_seq)
            loss = criterion(output, target_action)
            # print('output is', output)
            # print('target is ', target_action)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")



    end_time = time.time()
    time_to_retrain = end_time - start_time
    print(f'took {time_to_retrain} sec to retrain agent')
    save_model(agent_retrained_path, agent,optimizer)

    return agent

def get_str_obs(obs_st):
    if obs_st is None:
        return 'None'
    elif type(obs_st) is str:
        return obs_st
    elif obs_st.quality is None:
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

def get_team_action_from_single_action(action_name,agent_num):
    if checkIfCollabPush(action_name):
        return action_name+ 'X' + action_name
    if agent_num ==1:
        return action_name+'Xidle'
    else:
        return 'idleX'+action_name


def in_exit_area(n, pos):
    for i in range(0,len(pos),2):
        if pos[i]!=0 or pos[i+1] != n-1:
            return False
    return True

def get_action_traces_for_sync(agent_dict):
    agent_actions_sync_list=[]
    for num,trace_dict in agent_dict.items():
        tr = []
        for i in range(len(trace_dict['actions'])):
            agent_act = which_agent_operate_now(trace_dict['actions'][i])
            if agent_act ==1:
                tr.append('agent1')
            elif agent_act ==2:
                tr.append('agent2')
            else:
                tr.append('both')
        agent_actions_sync_list.append(tr)
    return agent_actions_sync_list


def get_maximal_intervals(trace_list):
    ret_list = []
    graph = dict()
    first_agent = trace_list[0][0]
    graph[0]=[first_agent,0]
    for i in range(len(trace_list)):
        interval = 0
        interval_counter =0
        for j in range(len(trace_list[i])):
            if graph[interval][0] == trace_list[i][j]:
                interval_counter +=1
                graph[interval][1]= max(interval_counter,graph[interval][1])
            else:
                interval+=1
                interval_counter= 1
                if interval not in graph.keys():
                    graph[interval] = [trace_list[i][j], interval_counter]
    for i in range(len(graph.items())):
        ret_list.extend([graph[i][0]] * (graph[i][1] + 1))  # plus 1 is for the idle i keep
        """if graph[i][0] =='both':
            ret_list.extend([graph[i][0]] * (graph[i][1]))
        elif i <len(graph.items()) and graph[i+1][0] == 'both':
            ret_list.extend([graph[i][0]] * (graph[i][1]))
        else:
            ret_list.extend([graph[i][0]] * (graph[i][1]+1) )# plus 1 is for the idle i keep"""
    return ret_list


def init_box_pushing_loadpolicy(n,k,number_of_agents,agents_init_loc,init_box_locations_lst,box_types,prob_to_push):
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
    return init_bf,true_init_bf, policy

def add_results_to_table_box_pushing(trace_number,init_state,agents_windows,terminal_state_flag,n,k,reward_ex,box_types,end_state,agents_results):

    agents_results[trace_number]=dict()
    agents_results[trace_number]['init_state'] = init_state
    agents_results[trace_number]['reach_goal']= terminal_state_flag
    steps =0
    check_count = 0
    move_count =0
    push_count = 0
    cpush_count = 0
    reward = 0
    for i in range(len(box_types)):
        if box_types[i] == 'S':
            reward+=500
        else:
            reward+=1000

    for agent_window in agents_windows:
        for i in range(len(agent_window)):
            if agent_window[i][2] != 'idle':
                steps+=1
                if agent_window[i][2] in {'left','right','up','down'}:
                    move_count+=1
                elif agent_window[i][2] in {'check1','check2','check3','check4'}:
                    check_count+=1
                elif agent_window[i][2][:-1] in {'pushUp', 'pushDown', 'pushLeft', 'pushRight'}:
                    push_count+=1
                elif agent_window[i][2][1:-1] in {'pushUp', 'pushDown', 'pushLeft', 'pushRight'}:
                    cpush_count+=1
    if terminal_state_flag == False:
        for i in range(0,len(end_state.box_locations)-1,2):
            loc_box = (end_state.box_locations[i],end_state.box_locations[i+1])
            if loc_box[0] != 0  or loc_box[1] != n-1:
                if box_types[int(i/2)] == 'S':
                    reward -= 500
                else:
                    reward -= 1000
    reward -= check_count * 1
    reward -= move_count * 10
    reward -= push_count * 30
    reward -= (cpush_count / 2) * 20


    agents_results[trace_number]['steps'] = steps
    agents_results[trace_number]['move_count'] = move_count
    agents_results[trace_number]['check_count'] = check_count
    agents_results[trace_number]['push_count'] = push_count
    agents_results[trace_number]['cpush_count'] = cpush_count / 2
    agents_results[trace_number]['reward'] = reward
    agents_results[trace_number]['reward_from_ex'] = reward_ex
    agents_results[trace_number]['end_state'] = end_state
def execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_types,prob_to_push,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path,sync_list):
    collab_push_actions = ['CpushUp', 'CpushDown', 'CpushLeft', 'CpushRight']
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model,_ = load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model,_= load_model(agent2_model_path)
    trace_i=0
    agents_results=dict()
    for i in range(0,1):
        for init_state in true_init_bf:
            total_reward = 0
            step = 0
            agent1_step = 0
            agent2_step = 0
            terminal_state_flag = False
            print(f'trace number: {trace_i} init state is : {init_state}')
            bp_problem = BoxPushginSampleProblem(n, k, init_state, box_types, number_of_agents, prob_to_push,
                                                 pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(bp_problem.agent)
            reward = bp_problem.env.state_transition(action, execute=True)
            total_reward+=reward
            observation = bp_problem.agent.observation_model.sample(bp_problem.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                          action, observation,
                                                          bp_problem.agent.observation_model,
                                                          bp_problem.agent.transition_model)
            bp_problem.agent.set_belief(new_belief)
            terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'

            #building agent windows
            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
            action_name = action.name
            action_arr = action_name.split('X')
            agent1_action = action_arr[0]
            agent2_action = action_arr[1]
            agent1_obs = 'None'
            agent2_obs = 'None'
            if which_agent_operate_now(action_name) == 1:
                agent1_obs = get_str_obs(observation.quality)
            else:
                agent2_obs = get_str_obs(observation.quality)
            agent1_window = [(agent1_step,agent1_loc,agent1_action,agent1_obs)]
            agent2_window = [(agent2_step,agent2_loc,agent2_action,agent2_obs)]
            agent1_didnt_do_idle = agent1_action != 'idle'
            agent2_didnt_do_idle = agent2_action != 'idle'
            agent1_step+=1
            agent2_step+=1
            step += 1
            while step<len(sync_list) and terminal_state_flag == False:
                if sync_list[step] == 'agent1':
                    agent2_didnt_do_idle =True
                    if agent1_didnt_do_idle == True:
                        # agent 1 turn -----------------------
                        agent1_emb_seq = embedd_seq(agent1_window, agent1_dict['locations'], agent1_dict['action_set'],
                                                    agent1_dict['observation_set'])
                        agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                        agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                        agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                        agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                        agent2_suppose = 'None_action'
                        if agent1_predicted_action_name[:-1] in collab_push_actions:
                            agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'],agent2_dict['action_set'], agent2_dict['observation_set'])
                            agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                            agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                            agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                            agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                            agent2_suppose = agent2_predicted_action_name
                        if agent1_predicted_action_name[:-1] in collab_push_actions and agent2_suppose == agent1_predicted_action_name:
                            d= 'skip'
                        elif agent1_predicted_action_name != 'idle':
                            action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name, 1)
                            action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                            reward_after_execute1 = bp_problem.env.state_transition(action_to_execute_for_env,
                                                                                            execute=True)
                            total_reward+=reward_after_execute1
                            observation_after_execute = bp_problem.agent.observation_model.sample(
                                bp_problem.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          bp_problem.agent.observation_model,
                                                                          bp_problem.agent.transition_model)
                            bp_problem.agent.set_belief(new_belief)
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs(observation_after_execute.quality)
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step+=1

                        else:
                            agent1_didnt_do_idle =False
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1


                elif sync_list[step] == 'agent2': #agent2_turn
                    agent1_didnt_do_idle = True
                    if agent2_didnt_do_idle == True:
                        #agent 2 turn -----------------------------------------------------------------------------------------------------------
                        agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'], agent2_dict['action_set'], agent2_dict['observation_set'])
                        agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                        agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                        agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                        agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                        agent1_suppose = 'None_action'
                        if agent2_predicted_action_name[:-1] in collab_push_actions:
                            agent1_emb_seq = embedd_seq(agent1_window, agent1_dict['locations'],
                                                        agent1_dict['action_set'], agent1_dict['observation_set'])
                            agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                            agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                            agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                            agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                            agent1_suppose = agent1_predicted_action_name
                        if agent2_predicted_action_name[:-1] in collab_push_actions and agent1_suppose == agent2_predicted_action_name:
                            d= 'skip'
                        elif agent2_predicted_action_name != 'idle':
                            action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name,2)
                            action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                            reward_after_execute2 = bp_problem.env.state_transition(action_to_execute_for_env, execute=True)
                            total_reward += reward_after_execute2
                            observation_after_execute = bp_problem.agent.observation_model.sample(bp_problem.env.state,
                                                                                            action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                          action_to_execute_for_env, observation_after_execute,
                                                                          bp_problem.agent.observation_model,
                                                                          bp_problem.agent.transition_model)
                            bp_problem.agent.set_belief(new_belief)
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs(observation_after_execute.quality)
                            agent2_window.append((agent2_step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                            agent2_step += 1
                        else:
                            agent2_didnt_do_idle = False
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                            agent2_step+=1
                            #agent 2 end turn --------------------------------------------------------------
                elif sync_list[step] == 'both':
                    agent1_didnt_do_idle = True
                    agent2_didnt_do_idle = True
                    agent1_emb_seq = embedd_seq(agent1_window, agent1_dict['locations'], agent1_dict['action_set'],agent1_dict['observation_set'])
                    agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                    agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                    agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                    agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                    agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'], agent2_dict['action_set'],agent2_dict['observation_set'])
                    agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                    agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                    agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                    agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                    action_to_execute = agent1_predicted_action_name + 'X' + agent2_predicted_action_name
                    if checkIfCollabPush(action_to_execute):

                        action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                        reward_after_execute = bp_problem.env.state_transition(action_to_execute_for_env, execute=True)
                        total_reward += reward_after_execute
                        observation_after_execute = bp_problem.agent.observation_model.sample(bp_problem.env.state,
                                                                                              action_to_execute_for_env)
                        new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                      action_to_execute_for_env,
                                                                      observation_after_execute,
                                                                      bp_problem.agent.observation_model,
                                                                      bp_problem.agent.transition_model)
                        bp_problem.agent.set_belief(new_belief)
                        agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                        agent1_obs = get_str_obs(observation_after_execute.quality)
                        agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                        agent1_step += 1

                        agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                        agent2_obs = get_str_obs(observation_after_execute.quality)
                        agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                        agent2_step += 1
                    if agent1_predicted_action_name == 'idle' and agent2_predicted_action_name == 'idle':
                        agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                        agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, 'None'))
                        agent1_step += 1
                        agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                        agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, 'None'))
                        agent2_step += 1

                terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'
                step+=1
            trace_i+=1
            print(f'success: {terminal_state_flag} in  {step} steps')
            print(agent1_window)
            print(agent2_window)
            print()
            add_results_to_table_box_pushing(trace_i,init_state,[agent1_window,agent2_window],terminal_state_flag,n,k,total_reward,box_types,bp_problem.env.cur_state,agents_results)
    return agents_results

def predict_action_for_trace(agent_model, agent_window,agent_dict):
    agent_emb_seq = embedd_seq(agent_window, agent_dict['locations'], agent_dict['action_set'],
                                agent_dict['observation_set'])
    agent_padded_seq = pad_seq(agent_emb_seq, agent_dict['max_len_in_train'])
    agent_input_sequence = torch.tensor(agent_padded_seq, dtype=torch.float32)
    agent_predicted_action = predict_next_action(agent_model, agent_input_sequence)
    agent_predicted_action_name = agent_dict['rev_action_set'][agent_predicted_action]
    return agent_predicted_action_name


def execute_for_box_push_loadpolicy_tranformer_sync_final(init_bf,true_init_bf,policy,n,k,box_types,prob_to_push,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path,sync_list):
    collab_push_actions = ['CpushUp', 'CpushDown', 'CpushLeft', 'CpushRight']
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model,_ = load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model,_= load_model(agent2_model_path)
    trace_i=0
    agents_results=dict()
    for i in range(0,1):
        for init_state in true_init_bf:
            total_reward = 0
            step = 0
            agent1_step = 0
            agent2_step = 0
            print(f'trace number: {trace_i} init state is : {init_state}')
            bp_problem = BoxPushginSampleProblem(n, k, init_state, box_types, number_of_agents, prob_to_push,
                                                 pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(bp_problem.agent)
            reward = bp_problem.env.state_transition(action, execute=True)
            total_reward+=reward
            observation = bp_problem.agent.observation_model.sample(bp_problem.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                          action, observation,
                                                          bp_problem.agent.observation_model,
                                                          bp_problem.agent.transition_model)
            bp_problem.agent.set_belief(new_belief)
            terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'


            #building agent windows
            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
            action_name = action.name
            action_arr = action_name.split('X')
            agent1_action = action_arr[0]
            agent2_action = action_arr[1]
            agent1_obs = 'None'
            agent2_obs = 'None'
            if which_agent_operate_now(action_name) == 1:
                agent1_obs = get_str_obs(observation.quality)
            else:
                agent2_obs = get_str_obs(observation.quality)
            agent1_window = [(agent1_step,agent1_loc,agent1_action,agent1_obs)]
            agent2_window = [(agent2_step,agent2_loc,agent2_action,agent2_obs)]
            agent1_step+=1
            agent2_step+=1
            step += 1
            while step<len(sync_list) and terminal_state_flag == False:
                if sync_list[step] == 'agent1':
                    agent1_predicted_action_name = predict_action_for_trace(agent1_model, agent1_window, agent1_dict)
                    if agent1_predicted_action_name == 'idle':
                        if agent1_window[-1][2] != 'idle':
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1
                        else:
                            print('bug more than one idle in trace sol in dec_tiger')
                    else:
                        if agent1_predicted_action_name[:-1] in collab_push_actions:
                            print('skip until both')
                        else:
                            action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name, 1)
                            action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                            reward_after_execute1 = bp_problem.env.state_transition(action_to_execute_for_env,
                                                                                    execute=True)
                            total_reward += reward_after_execute1
                            observation_after_execute = bp_problem.agent.observation_model.sample(
                                bp_problem.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          bp_problem.agent.observation_model,
                                                                          bp_problem.agent.transition_model)
                            bp_problem.agent.set_belief(new_belief)
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs(observation_after_execute.quality)
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1
                elif sync_list[step] == 'agent2':
                    agent2_predicted_action_name = predict_action_for_trace(agent2_model, agent2_window, agent2_dict)
                    if agent2_predicted_action_name == 'idle':
                        if agent2_window[-1][2] != 'idle':
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                            agent2_step += 1
                        else:
                            print('bug more than one idle in trace sol in dec_tiger')
                    else:
                        if agent2_predicted_action_name[:-1] in collab_push_actions:
                            print('skip until both')
                        else:
                            action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name, 2)
                            action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                            reward_after_execute2 = bp_problem.env.state_transition(action_to_execute_for_env,
                                                                                    execute=True)
                            total_reward += reward_after_execute2
                            observation_after_execute = bp_problem.agent.observation_model.sample(
                                bp_problem.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          bp_problem.agent.observation_model,
                                                                          bp_problem.agent.transition_model)
                            bp_problem.agent.set_belief(new_belief)
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs(observation_after_execute)
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                            agent2_step += 1
                elif sync_list[step] == 'both':
                    agent1_predicted_action_name = predict_action_for_trace(agent1_model,agent1_window,agent1_dict)
                    agent2_predicted_action_name = predict_action_for_trace(agent2_model,agent2_window,agent2_dict)
                    action_to_execute = agent1_predicted_action_name + 'X' + agent2_predicted_action_name
                    if checkIfCollabPush(action_to_execute):
                        action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                        reward_after_execute = bp_problem.env.state_transition(action_to_execute_for_env, execute=True)
                        total_reward += reward_after_execute
                        observation_after_execute = bp_problem.agent.observation_model.sample(bp_problem.env.state,
                                                                                              action_to_execute_for_env)
                        new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                      action_to_execute_for_env,
                                                                      observation_after_execute,
                                                                      bp_problem.agent.observation_model,
                                                                      bp_problem.agent.transition_model)
                        bp_problem.agent.set_belief(new_belief)
                        agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                        agent1_obs = get_str_obs(observation_after_execute)
                        agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                        agent1_step += 1

                        agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                        agent2_obs = get_str_obs(observation_after_execute)
                        agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                        agent2_step += 1
                    else:
                        if agent1_predicted_action_name[:-1] in collab_push_actions:
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1
                        if agent2_predicted_action_name[:-1] in collab_push_actions:
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                            agent2_step += 1
                    if agent1_predicted_action_name == 'idle' and agent2_predicted_action_name == 'idle':
                        if agent1_window[-1][2] != 'idle':
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, 'None'))
                            agent1_step += 1
                        if agent2_window[-1][2] != 'idle':
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, 'None'))
                            agent2_step += 1
                terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'
                step+=1
            trace_i += 1
            print(f'success: {terminal_state_flag} in  {step} steps')
            print('total reward ', total_reward)
            print(agent1_window)
            print(agent2_window)
            print()
            add_results_to_table_box_pushing(trace_i,init_state,[agent1_window,agent2_window],terminal_state_flag,n,k,total_reward,box_types,bp_problem.env.cur_state,agents_results)
    return agents_results
def teacher_correction(agent1_window,agent2_window,belief_state,curr_real_state,policy):
    check_point_in_trace = dict()
    check_point_in_trace['agent1_window'] = agent1_window
    check_point_in_trace['agent2_window'] = agent2_window
    check_point_in_trace['belief_state'] = belief_state
    check_point_in_trace['state'] = curr_real_state
    return check_point_in_trace

def execute_for_box_push_loadpolicy_tranformer_policyrefine(init_bf,true_init_bf,policy,n,k,box_types,prob_to_push,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path,sync_list):
    collab_push_actions = ['CpushUp', 'CpushDown', 'CpushLeft', 'CpushRight']
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model,_ = load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model,_= load_model(agent2_model_path)
    trace_i=0
    traces_to_correct = dict()
    index_for_traces_to_correct = 0
    for i in range(0,1):
        for init_state in true_init_bf:
            curr_trace_to_correct=[]
            total_reward = 0
            step = 0
            agent1_step = 0
            agent2_step = 0
            print(f'trace number: {trace_i} init state is : {init_state}')
            bp_problem = BoxPushginSampleProblem(n, k, init_state, box_types, number_of_agents, prob_to_push,
                                                 pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(bp_problem.agent)
            reward = bp_problem.env.state_transition(action, execute=True)
            total_reward+=reward
            observation = bp_problem.agent.observation_model.sample(bp_problem.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                          action, observation,
                                                          bp_problem.agent.observation_model,
                                                          bp_problem.agent.transition_model)
            bp_problem.agent.set_belief(new_belief)
            terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'


            #building agent windows
            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
            action_name = action.name
            action_arr = action_name.split('X')
            agent1_action = action_arr[0]
            agent2_action = action_arr[1]
            agent1_obs = 'None'
            agent2_obs = 'None'
            if which_agent_operate_now(action_name) == 1:
                agent1_obs = get_str_obs(observation.quality)
            else:
                agent2_obs = get_str_obs(observation.quality)
            agent1_window = [(agent1_step,agent1_loc,agent1_action,agent1_obs)]
            agent2_window = [(agent2_step,agent2_loc,agent2_action,agent2_obs)]
            agent1_step+=1
            agent2_step+=1
            step += 1
            while step<len(sync_list) and terminal_state_flag == False:
                if sync_list[step] == 'agent1':
                    agent1_predicted_action_name = predict_action_for_trace(agent1_model, agent1_window, agent1_dict)
                    if agent1_predicted_action_name == 'idle':
                        if agent1_window[-1][2] != 'idle':
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1
                        else:
                            print('bug more than one idle in trace sol in dec_tiger')
                    else:
                        if agent1_predicted_action_name[:-1] in collab_push_actions:
                            print('skip until both')
                        else:
                            action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name, 1)
                            action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                            action_from_teacher = policy.plan(bp_problem.agent)
                            if not checkIfCheckAction(action_to_execute) and  action_from_teacher != action_to_execute:
                                temp_res=teacher_correction(agent1_window,agent2_window,bp_problem.agent.cur_belief,bp_problem.env.state,policy)
                                curr_trace_to_correct.append(temp_res)
                            reward_after_execute1 = bp_problem.env.state_transition(action_to_execute_for_env,
                                                                                    execute=True)
                            total_reward += reward_after_execute1
                            observation_after_execute = bp_problem.agent.observation_model.sample(
                                bp_problem.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          bp_problem.agent.observation_model,
                                                                          bp_problem.agent.transition_model)
                            bp_problem.agent.set_belief(new_belief)
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs(observation_after_execute.quality)
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1
                elif sync_list[step] == 'agent2':
                    agent2_predicted_action_name = predict_action_for_trace(agent2_model, agent2_window, agent2_dict)
                    if agent2_predicted_action_name == 'idle':
                        if agent2_window[-1][2] != 'idle':
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                            agent2_step += 1
                        else:
                            print('bug more than one idle in trace sol in dec_tiger')
                    else:
                        if agent2_predicted_action_name[:-1] in collab_push_actions:
                            print('skip until both')
                        else:
                            action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name, 2)
                            action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                            action_from_teacher = policy.plan(bp_problem.agent)
                            if not checkIfCheckAction(action_to_execute) and  action_from_teacher != action_to_execute:
                                temp_res=teacher_correction(agent1_window,agent2_window,bp_problem.agent.cur_belief,bp_problem.env.state,policy)
                                curr_trace_to_correct.append(temp_res)
                            reward_after_execute2 = bp_problem.env.state_transition(action_to_execute_for_env,
                                                                                    execute=True)
                            total_reward += reward_after_execute2
                            observation_after_execute = bp_problem.agent.observation_model.sample(
                                bp_problem.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          bp_problem.agent.observation_model,
                                                                          bp_problem.agent.transition_model)
                            bp_problem.agent.set_belief(new_belief)
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs(observation_after_execute)
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                            agent2_step += 1
                elif sync_list[step] == 'both':
                    agent1_predicted_action_name = predict_action_for_trace(agent1_model,agent1_window,agent1_dict)
                    agent2_predicted_action_name = predict_action_for_trace(agent2_model,agent2_window,agent2_dict)
                    action_to_execute = agent1_predicted_action_name + 'X' + agent2_predicted_action_name
                    action_from_teacher = policy.plan(bp_problem.agent)
                    if action_to_execute !='idleXidle' and action_from_teacher != action_to_execute:
                        temp_res = teacher_correction(agent1_window, agent2_window, bp_problem.agent.cur_belief,
                                                      bp_problem.env.state, policy)
                        curr_trace_to_correct.append(temp_res)
                    if checkIfCollabPush(action_to_execute):
                        action_to_execute_for_env = BoxPushingSampleAction(action_to_execute)
                        reward_after_execute = bp_problem.env.state_transition(action_to_execute_for_env, execute=True)
                        total_reward += reward_after_execute
                        observation_after_execute = bp_problem.agent.observation_model.sample(bp_problem.env.state,
                                                                                              action_to_execute_for_env)
                        new_belief = pomdp_py.update_histogram_belief(bp_problem.agent.cur_belief,
                                                                      action_to_execute_for_env,
                                                                      observation_after_execute,
                                                                      bp_problem.agent.observation_model,
                                                                      bp_problem.agent.transition_model)
                        bp_problem.agent.set_belief(new_belief)
                        agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                        agent1_obs = get_str_obs(observation_after_execute)
                        agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                        agent1_step += 1

                        agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                        agent2_obs = get_str_obs(observation_after_execute)
                        agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                        agent2_step += 1
                    else:
                        if agent1_predicted_action_name[:-1] in collab_push_actions:
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1
                        if agent2_predicted_action_name[:-1] in collab_push_actions:
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, agent2_obs))
                            agent2_step += 1
                    if agent1_predicted_action_name == 'idle' and agent2_predicted_action_name == 'idle':
                        if agent1_window[-1][2] != 'idle':
                            agent1_loc = (bp_problem.env.cur_state.position[0], bp_problem.env.cur_state.position[1])
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, 'None'))
                            agent1_step += 1
                        if agent2_window[-1][2] != 'idle':
                            agent2_loc = (bp_problem.env.cur_state.position[2], bp_problem.env.cur_state.position[3])
                            agent2_window.append((agent2_step, agent2_loc, agent2_predicted_action_name, 'None'))
                            agent2_step += 1
                terminal_state_flag = bp_problem.env.cur_state.position[0] == 'T'
                step+=1
            trace_i += 1
            print(f'success: {terminal_state_flag} in  {step} steps')
            if terminal_state_flag is False:
                traces_to_correct[index_for_traces_to_correct] = curr_trace_to_correct
                index_for_traces_to_correct+=1
            print('total reward ', total_reward)
            print(agent1_window)
            print(agent2_window)
            print()
            #add_results_to_table_box_pushing(trace_i,init_state,[agent1_window,agent2_window],terminal_state_flag,n,k,total_reward,box_types,bp_problem.env.cur_state,agents_results)
    return traces_to_correct

if __name__ == '__main__':
    print('boxpush trans hello')
    date_for_traces='10_08_2023'
    date= '10_08_2023'
    n = 2
    k = 2
    number_of_agents = 2
    prob_for_push = 0.8
    box_type = ['L','L']
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

    init_bf,true_init_bf,policy = init_box_pushing_loadpolicy(n,k,number_of_agents,init_agents_location,init_box_location_flatten,box_type,prob_for_push)
    action_map = {'idle': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}
    index_map = 5
    single_agent_push = ['pushUp', 'pushDown', 'pushLeft', 'pushRight']
    for i in range(len(box_type)):
        check_str = 'check' + str(i+1)
        action_map[check_str] = index_map
        index_map += 1
        for push_action in single_agent_push:
            str_to_add = push_action +str(i+1)
            if box_type[i] == 'L':
                str_to_add = "C" +str_to_add
            action_map[str_to_add]=index_map
            index_map+=1


    observation_map = {'None': 0, 'Y': 1, 'N': 2}


    team_path = f'traces/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_team.pickle'
    agent1_path = f'traces/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_agent1.pickle'
    agent2_path = f'traces/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_agent2.pickle'
    agent1_bad_trace_path = f'traces/box_pushing_bad_traces{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_agent1.pickle'
    agent2_bad_trace_path = f'traces/box_pushing_bad_traces{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_agent2.pickle'
    team_retrain_path = f'traces/box_pushing_retrain{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_team.pickle'
    agent1_retrain_traces_path = f'traces/box_pushing_retrain_traces{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_agent1.pickle'
    agent2_retrain_traces_path = f'traces/box_pushing_retrain_traces{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date_for_traces}_agent2.pickle'
    agent1_save_model_path = f'transformer_saves/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}_agent1.pth'
    retrained_agent1_save_model_path = f'transformer_saves/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}_agent1_retrain.pth'
    agent1_save_essential_path = f'transformer_saves/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}_agent1_essential.pickle'
    agent2_save_model_path = f'transformer_saves/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}_agent2.pth'
    retrained_agent2_save_model_path = f'transformer_saves/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}_agent2_retrain.pth'
    agent2_save_essential_path = f'transformer_saves/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}_agent2_essential.pickle'
    team_sync_path = f'traces/box_pushing_sync_lst{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}.pickle'
    results_path = f'results/box_pushing{n}x{k}x{number_of_agents}x{countSmall}x{countLarge}_{date}.pickle'
    phase = 5
    if phase ==1:# train agents for the first time with sync
        team_dict = load_and_preprocess(team_path)
        traces= get_action_traces_for_sync(team_dict)
        result = get_maximal_intervals(traces)
        dict_result = {'sync_list':result}
        save_dict(team_sync_path,dict_result)
        agent1_dict = load_and_preprocess(agent1_path)
        agent2_dict = load_and_preprocess(agent2_path)
        add_idle_for_collab = True
        agent1_traces = get_traces_only_one_idle_dispatch(agent1_dict,add_idle_for_collab)
        agent2_traces = get_traces_only_one_idle_dispatch(agent2_dict, add_idle_for_collab)
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)

        train_agent(agent1_dict,agent1_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path,n,k,action_map,observation_map)

        train_agent(agent2_dict,agent2_traces,max_len_agent2,agent2_save_model_path,agent2_save_essential_path,n,k,action_map,observation_map)

    if phase ==2: #check trained agents synced
        result_from_sync = load_and_preprocess(team_sync_path)
        result_from_sync = result_from_sync['sync_list']
        start_time = time.time()
        #results=execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_type,prob_for_push,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
        results = execute_for_box_push_loadpolicy_tranformer_sync_final(init_bf, true_init_bf, policy, n, k, box_type,
                                                                  prob_for_push, agent1_save_model_path,
                                                                  agent1_save_essential_path, agent2_save_model_path,
                                                                  agent2_save_essential_path, result_from_sync)
        end_time = time.time()
        time_to_detect_bad_traces = end_time - start_time
        print(f'took {time_to_detect_bad_traces} sec to produce traces')
        #save_dict(results_path,results)
        tries = len(results.keys())
        sum_reward = 0
        sum_steps= 0
        succ_rate = 0
        for i,trace_res in results.items():
            print(trace_res)
            sum_steps+=trace_res['steps']
            max_reward = max(trace_res['reward'],trace_res['reward_from_ex'])
            sum_reward+= trace_res['reward']
            if trace_res['reach_goal'] ==True:
                succ_rate+=1
        print(sum_steps/tries)
        print(sum_reward / tries)
        print(succ_rate / tries)
        print('DONE')

    if phase == 3:  # refine policy
        result_from_sync = load_and_preprocess(team_sync_path)
        result_from_sync = result_from_sync['sync_list']
        start_time = time.time()
        # results=execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_type,prob_for_push,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
        traces_for_correction = execute_for_box_push_loadpolicy_tranformer_policyrefine(init_bf, true_init_bf, policy, n, k,
                                                                        box_type,
                                                                        prob_for_push, agent1_save_model_path,
                                                                        agent1_save_essential_path,
                                                                        agent2_save_model_path,
                                                                        agent2_save_essential_path,
                                                                        result_from_sync)
        genrate_traces_for_retrain(traces_for_correction,policy,agent1_retrain_traces_path,agent2_retrain_traces_path,team_retrain_path,n,k,box_type,number_of_agents,0.8)

    if phase == 4: #retrain
        agent1_dict = load_and_preprocess(agent1_retrain_traces_path)
        agent2_dict = load_and_preprocess(agent2_retrain_traces_path)
        add_idle_for_collab = True
        agent1_traces = get_traces_only_one_idle_dispatch(agent1_dict,add_idle_for_collab)
        agent2_traces = get_traces_only_one_idle_dispatch(agent2_dict, add_idle_for_collab)
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)
        agen1_model_retrained = retrain_agent(agent1_traces,agent1_save_model_path,agent1_save_essential_path,retrained_agent1_save_model_path)
        agen2_model_retrained = retrain_agent(agent2_traces, agent2_save_model_path, agent2_save_essential_path,retrained_agent2_save_model_path)
    if phase == 5: #check trained agents synced
        result_from_sync = load_and_preprocess(team_sync_path)
        result_from_sync = result_from_sync['sync_list']
        start_time = time.time()
        #results=execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_type,prob_for_push,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
        results = execute_for_box_push_loadpolicy_tranformer_sync_final(init_bf, true_init_bf, policy, n, k, box_type,
                                                                  prob_for_push, retrained_agent1_save_model_path,
                                                                  agent1_save_essential_path, retrained_agent2_save_model_path,
                                                                  agent2_save_essential_path, result_from_sync)
        end_time = time.time()
        time_to_detect_bad_traces = end_time - start_time
        print(f'took {time_to_detect_bad_traces} sec to produce traces')
        #save_dict(results_path,results)
        tries = len(results.keys())
        sum_reward = 0
        sum_steps= 0
        succ_rate = 0
        for i,trace_res in results.items():
            print(trace_res)
            sum_steps+=trace_res['steps']
            max_reward = max(trace_res['reward'],trace_res['reward_from_ex'])
            sum_reward+= trace_res['reward']
            if trace_res['reach_goal'] ==True:
                succ_rate+=1
        print(sum_steps/tries)
        print(sum_reward / tries)
        print(succ_rate / tries)
        print('DONE')



        if phase == 6: 
            team_dict = load_and_preprocess(team_path)
            traces= get_action_traces_for_sync(team_dict)
            result = get_maximal_intervals(traces)
            dict_result = {'sync_list':result}
            save_dict(team_sync_path,dict_result)
            agent1_dict = load_and_preprocess(agent1_path)
            agent2_dict = load_and_preprocess(agent2_path)
            add_idle_for_collab = True
            agent1_traces = get_traces_only_one_idle_dispatch(agent1_dict,add_idle_for_collab)
            agent2_traces = get_traces_only_one_idle_dispatch(agent2_dict, add_idle_for_collab)
            max_len_agent1 = get_max_len(agent1_traces)
            max_len_agent2 = get_max_len(agent2_traces)

            train_agent(agent1_dict,agent1_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path,n,k,action_map,observation_map)

            train_agent(agent2_dict,agent2_traces,max_len_agent2,agent2_save_model_path,agent2_save_essential_path,n,k,action_map,observation_map)

            #----------------------------refine# 
            result_from_sync = load_and_preprocess(team_sync_path)
            result_from_sync = result_from_sync['sync_list']
            start_time = time.time()
            # results=execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_type,prob_for_push,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
            traces_for_correction = execute_for_box_push_loadpolicy_tranformer_policyrefine(init_bf, true_init_bf, policy, n, k,
                                                                            box_type,
                                                                            prob_for_push, agent1_save_model_path,
                                                                            agent1_save_essential_path,
                                                                            agent2_save_model_path,
                                                                            agent2_save_essential_path,
                                                                            result_from_sync)
            team_traces=genrate_traces_for_retrain(traces_for_correction,policy,agent1_retrain_traces_path,agent2_retrain_traces_path,team_retrain_path,n,k,box_type,number_of_agents,0.8)
            max_refine_steps = 4
            i = 0
            while(i<max_refine_steps and len(team_traces>0) ):
                agent1_dict = load_and_preprocess(agent1_retrain_traces_path)
                agent2_dict = load_and_preprocess(agent2_retrain_traces_path)
                add_idle_for_collab = True
                agent1_traces = get_traces_only_one_idle_dispatch(agent1_dict,add_idle_for_collab)
                agent2_traces = get_traces_only_one_idle_dispatch(agent2_dict, add_idle_for_collab)
                max_len_agent1 = get_max_len(agent1_traces)
                max_len_agent2 = get_max_len(agent2_traces)
                agen1_model_retrained = retrain_agent(agent1_traces,agent1_save_model_path,agent1_save_essential_path,retrained_agent1_save_model_path)
                agen2_model_retrained = retrain_agent(agent2_traces, agent2_save_model_path, agent2_save_essential_path,retrained_agent2_save_model_path)
                #-------------#
                result_from_sync = load_and_preprocess(team_sync_path)
                result_from_sync = result_from_sync['sync_list']
                start_time = time.time()
                # results=execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_type,prob_for_push,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
                traces_for_correction = execute_for_box_push_loadpolicy_tranformer_policyrefine(init_bf, true_init_bf, policy, n, k,
                                                                                box_type,
                                                                                prob_for_push, agent1_save_model_path,
                                                                                agent1_save_essential_path,
                                                                                agent2_save_model_path,
                                                                                agent2_save_essential_path,
                                                                                result_from_sync)
                team_traces=genrate_traces_for_retrain(traces_for_correction,policy,agent1_retrain_traces_path,agent2_retrain_traces_path,team_retrain_path,n,k,box_type,number_of_agents,0.8)
                i+=1