from pomdp_problems.tiger.tiger_problem import TigerProblem , TigerState
from pomdp_problems.dec_tiger.dec_tiger_problem import DecTigerProblem, DecTigerState, DecTigerAction
from pomdp_problems.rocksample.rocksample_problem import RockSampleProblem, State, init_particles_belief
from pomdp_problems.dec_rocksample.dec_rocksample_problem import DecRockSampleProblem, DecRockSampleState, DecRockSampleAction
from pomdp_problems.small_dec_rock.small_dec_rock_problem import SmallDecRockSampleProblem, SmallDecRockSampleState, SmallDecRockSampleAction, SmallDecRockSampleObservation
from pomdp_problems.general_dec_rock.general_dec_rock_problem import GenDecRockSampleProblem, GenDecRockSampleState, GenDecRockSampleAction, GenDecRockSampleObservation, checkIfCheckAction
from pomdp_py import to_pomdp_file
from pomdp_py import to_pomdpx_file
from pomdp_py import vi_pruning
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import itertools

import pomdp_py
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from decRock_refine_traces import traces_for_retrain

from pomdp_problems.gen_final_dec_rock.gen_final_dec_rock_problem import GenDecRockSampleProblem, GenDecRockSampleState, GenDecRockSampleAction, GenDecRockSampleObservation, checkIfCheckAction
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

def load_and_preprocess(pickle_path):
    with open(pickle_path, 'rb') as handle:
        agentdict = pickle.load(handle)
    return agentdict
def save_dict(path,dict_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
def get_traces(agent_dict):
  agent_trajectories =[]
  for num,trace_dict in agent_dict.items():
    tr = []
    for i in range(len(trace_dict['states'])):
      tr.append((i,trace_dict['states'][i],trace_dict['actions'][i],trace_dict['observations'][i]))
    agent_trajectories.append(tr)
  return agent_trajectories
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


# Define the Transformer model
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


# Define the dataset class
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


def train_agent1(agent1_dict,agent1_traces,max_len_agent1,agent_model_path,essential_path):
    start_time = time.time()
    #def agent 1 ------------------------------------------------------------------
    #for 4x3x3
    #action_map = {'idle': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4, 'check1': 5, 'check2': 6, 'check3': 7,'sample1': 8, 'sample2': 9, 'sample3': 10}
    #for 4x3x4
    action_map = {'idle': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4, 'check1': 5, 'check2': 6, 'check3': 7,'sample1': 8, 'sample2': 9, 'sample3': 10}
    rev_actions = rev_vocab(action_map)
    observation_map = {'None': 0, 'G': 1, 'B': 2}
    rev_observations = rev_vocab(observation_map)
    locs = gen_map_for_locations(4, 5)
    rev_locs = rev_vocab(locs)
    essentials = {}
    essentials['action_set'] = action_map
    essentials['rev_action_set'] = rev_actions
    essentials['observation_set']  = observation_map
    essentials['rev_observation_set'] = rev_observations
    essentials['locations'] = locs
    essentials['rev_locations'] =rev_locs
    essentials['max_len_in_train'] = max_len_agent1
    #end def -----------------------------------------------------------------------

    input_dim = 4  # Dimensionality of the input (index,location, action, observation)
    output_dim = len(action_map.keys())  # Dimensionality of the output (number of actions)
    hidden_dim = 32  # Hidden dimension size
    num_layers = 3  # Number of transformer layers
    num_heads = 4  # Number of attention heads
    batch_size = 16
    learning_rate = 0.001
    epochs = 100
    # Create the Transformer agent
    agent = TransformerAgent(input_dim, output_dim, hidden_dim, num_layers, num_heads)

    # Create the dataset and data loader
    agent1_trajectories = agent1_traces
    agent1_all_trajectories = create_all_traj(agent1_trajectories)

    agent1_dataset = TrajectoryDataset(agent1_all_trajectories, locs, action_map, observation_map,max_len_agent1)
    dataloader = DataLoader(agent1_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
    """example_1 = agent1_traces[0]
    print(example_1)
    for i in range(3,len(example_1)):
        example_tr_1 = example_1[:i]
        print(example_tr_1)
        emb_seq = embedd_seq(example_tr_1, locs, action_map, observation_map)
        print(emb_seq)
        padded_seq = pad_seq(emb_seq, max_len_agent1)
        print(padded_seq)
        input_sequence = torch.tensor(padded_seq, dtype=torch.float32)
        predicted_action = predict_next_action(agent, input_sequence)
        print(f"Predicted action: {predicted_action} which is {rev_actions[predicted_action]}")"""

    end_time = time.time()
    time_to_train_agent1 = end_time - start_time
    print(f'took {time_to_train_agent1} sec to train agent1')
    save_model(agent_model_path, agent,optimizer)
    save_dict(essential_path,essentials)
    return agent

def train_agent2(agent_dict,agent_traces,max_len_agent,agent_model_path,essential_path):

    #def agent 2 ------------------------------------------------------------------
    #for 4x3x3
    #action_map = {'idle': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4,'check2': 5, 'check3': 6,'sample2': 7, 'sample3': 8}
    #for 4x3x4
    action_map = {'idle': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4,'check2': 5, 'check3': 6,'check4':7,'sample2': 8, 'sample3': 9, 'sample4': 10}
    rev_actions = rev_vocab(action_map)
    observation_map = {'None': 0, 'G': 1, 'B': 2}
    rev_observations = rev_vocab(observation_map)
    locs = gen_map_for_locations(4, 5)
    rev_locs = rev_vocab(locs)
    essentials = {}
    essentials['action_set'] = action_map
    essentials['rev_action_set'] = rev_actions
    essentials['observation_set']  = observation_map
    essentials['rev_observation_set'] = rev_observations
    essentials['locations'] = locs
    essentials['rev_locations'] =rev_locs
    essentials['max_len_in_train'] =max_len_agent
    #end def -----------------------------------------------------------------------

    input_dim = 4  # Dimensionality of the input (location, action, observation)
    output_dim = len(action_map.keys())  # Dimensionality of the output (number of actions)
    hidden_dim = 32  # Hidden dimension size
    num_layers = 3 # Number of transformer layers
    num_heads = 4  # Number of attention heads
    batch_size = 16
    learning_rate = 0.001
    epochs = 100
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
    """example_1 = agent1_traces[0]
    print(example_1)
    for i in range(3,len(example_1)):
        example_tr_1 = example_1[:i]
        print(example_tr_1)
        emb_seq = embedd_seq(example_tr_1, locs, action_map, observation_map)
        print(emb_seq)
        padded_seq = pad_seq(emb_seq, max_len_agent1)
        print(padded_seq)
        input_sequence = torch.tensor(padded_seq, dtype=torch.float32)
        predicted_action = predict_next_action(agent, input_sequence)
        print(f"Predicted action: {predicted_action} which is {rev_actions[predicted_action]}")"""


    save_model(agent_model_path, agent,optimizer)
    save_dict(essential_path, essentials)
    return agent

def retrain_agent(agent_dict,agent_traces,max_len_agent,agent_model_path,agent_essential_path,agent_retrained_path):


    agent_dict = load_and_preprocess(agent_essential_path)
    action_map = agent_dict['action_set']
    observation_map = agent_dict['observation_set']
    locs = agent_dict['locations']
    max_len_agent = agent_dict['max_len_in_train']
    start_time = time.time()


    input_dim = 4  # Dimensionality of the input (index,location, action, observation)
    output_dim = len(action_map.keys())  # Dimensionality of the output (number of actions)
    hidden_dim = 32  # Hidden dimension size
    num_layers = 2  # Number of transformer layers
    num_heads = 4  # Number of attention heads
    batch_size = 16
    learning_rate = 0.001
    epochs = 3
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

    "running example sane check"
    """example_1 = agent_traces[0]
    print(example_1)
    for i in range(3,len(example_1)):
        example_tr_1 = example_1[:i]
        print(example_tr_1)
        emb_seq = embedd_seq(example_tr_1, locs, action_map, observation_map)
        print(emb_seq)
        padded_seq = pad_seq(emb_seq, max_len_agent1)
        print(padded_seq)
        input_sequence = torch.tensor(padded_seq, dtype=torch.float32)
        predicted_action = predict_next_action(agent, input_sequence)
        print(f"Predicted action: {predicted_action} which is {agent_dict['rev_action_set'][predicted_action]}")"""

    end_time = time.time()
    time_to_retrain = end_time - start_time
    print(f'took {time_to_retrain} sec to retrain agent')
    save_model(agent_retrained_path, agent,optimizer)

    return agent
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



def gen_dec_get_all_states(_n,_k,specificBound,number_of_rocks):
    """Only need to implement this if you're using
    a solver that needs to enumerate over the observation space (e.g. value iteration)
    for now doing specific for 3x3 grid with col 1 as collab"""
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

def get_str_obs(obs_st):
    if obs_st is None:
        return 'None'
    else:
        return obs_st

def get_team_action_from_single_action(action_name,agent_num):
    if agent_num ==1:
        return action_name+'Xidle'
    else:
        return 'idleX'+action_name

def execute_for_dec_rock_loadpolicy_tranformer(n,k,bound,rock_loc_param,agents_init_loc,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path):
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
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model,_ = load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model,_= load_model(agent2_model_path)
    trace_i=0

    for i in range(0,1):
        for init_state in true_init_bf:
            step = 0
            terminal_state_flag = False
            print(f'trace number: {trace_i} init state is : {init_state}')
            gen_dec_rocksample = GenDecRockSampleProblem(n, k, bound, init_state, rocks_locs,
                                                         pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(gen_dec_rocksample.agent)
            reward = gen_dec_rocksample.env.state_transition(action, execute=True)
            observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                          action, observation,
                                                          gen_dec_rocksample.agent.observation_model,
                                                          gen_dec_rocksample.agent.transition_model)
            gen_dec_rocksample.agent.set_belief(new_belief)
            terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'

            #building agent windows
            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
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
            agent1_window = [(step,agent1_loc,agent1_action,agent1_obs)]
            agent2_window = [(step,agent2_loc,agent2_action,agent2_obs)]
            step += 1
            while step<20 and terminal_state_flag == False:
                #agent 1 turn -----------------------
                agent1_emb_seq = embedd_seq(agent1_window, agent1_dict['locations'], agent1_dict['action_set'], agent1_dict['observation_set'])
                #print(agent1_emb_seq)
                agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                #print(agent1_padded_seq)
                agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                #print(f"Predicted action: {agent1_predicted_action} which is {agent1_predicted_action_name}")
                if agent1_predicted_action_name != 'idle':
                    action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name,1)
                    action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                    reward_after_execute1 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env, execute=True)
                    observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_execute_for_env)
                    new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_execute_for_env, observation_after_execute,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                    gen_dec_rocksample.agent.set_belief(new_belief)
                    agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                    agent1_obs = get_str_obs(observation_after_execute.quality)
                    agent1_window.append((step,agent1_loc,agent1_predicted_action_name,agent1_obs))
                else:
                    agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                    agent1_obs = get_str_obs('None')
                    agent1_window.append((step,agent1_loc,agent1_predicted_action_name,agent1_obs))

                #agent1 end turn
                #agent 2 turn -----------------------------------------------------------------------------------------------------------
                agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'], agent2_dict['action_set'], agent2_dict['observation_set'])
                #print(agent2_emb_seq)
                agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                #print(agent2_padded_seq)
                agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                #print(f"Predicted action: {agent2_predicted_action} which is {agent2_predicted_action_name}")
                if agent2_predicted_action_name != 'idle':
                    action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name,2)
                    action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                    reward_after_execute2 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env, execute=True)
                    observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_execute_for_env)
                    new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_execute_for_env, observation_after_execute,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                    gen_dec_rocksample.agent.set_belief(new_belief)
                    agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                    agent2_obs = get_str_obs(observation_after_execute.quality)
                    agent2_window.append((step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                else:
                    agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                    agent2_obs = get_str_obs('None')
                    agent2_window.append((step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                #agent 2 end turn --------------------------------------------------------------
                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'
                step+=1
            trace_i+=1
            print(f'success: {terminal_state_flag} in  {step} steps')
            print(agent1_window)
            print(agent2_window)
            print()





def execute_for_dec_rock_loadpolicy_tranformer_sync(n,k,bound,rock_loc_param,agents_init_loc,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path,sync_list):
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
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model,_ = load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model,_= load_model(agent2_model_path)
    trace_i=0
    agents_results=dict()
    for i in range(0,1):
        for init_state in true_init_bf:
            step = 0
            agent1_step = 0
            agent2_step = 0
            terminal_state_flag = False
            print(f'trace number: {trace_i} init state is : {init_state}')
            gen_dec_rocksample = GenDecRockSampleProblem(n, k, bound, init_state, rocks_locs,
                                                         pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(gen_dec_rocksample.agent)
            reward = gen_dec_rocksample.env.state_transition(action, execute=True)
            observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                          action, observation,
                                                          gen_dec_rocksample.agent.observation_model,
                                                          gen_dec_rocksample.agent.transition_model)
            gen_dec_rocksample.agent.set_belief(new_belief)
            terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'

            #building agent windows
            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
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
                        # print(agent1_emb_seq)
                        agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                        # print(agent1_padded_seq)
                        agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                        agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                        agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                        # print(f"Predicted action: {agent1_predicted_action} which is {agent1_predicted_action_name}")

                        if agent1_predicted_action_name != 'idle':
                            action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name, 1)
                            action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                            reward_after_execute1 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env,
                                                                                            execute=True)
                            observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(
                                gen_dec_rocksample.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          gen_dec_rocksample.agent.observation_model,
                                                                          gen_dec_rocksample.agent.transition_model)
                            gen_dec_rocksample.agent.set_belief(new_belief)
                            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                            agent1_obs = get_str_obs(observation_after_execute.quality)
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step+=1

                        else:
                            agent1_didnt_do_idle =False
                            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1


                else: #agent2_turn
                    agent1_didnt_do_idle = True
                    if agent2_didnt_do_idle == True:
                        #agent 2 turn -----------------------------------------------------------------------------------------------------------
                        agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'], agent2_dict['action_set'], agent2_dict['observation_set'])
                        #print(agent2_emb_seq)
                        agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                        #print(agent2_padded_seq)
                        agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                        agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                        agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                        #print(f"Predicted action: {agent2_predicted_action} which is {agent2_predicted_action_name}")
                        if agent2_predicted_action_name != 'idle':
                            action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name,2)
                            action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                            reward_after_execute2 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env, execute=True)
                            observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                            action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                          action_to_execute_for_env, observation_after_execute,
                                                                          gen_dec_rocksample.agent.observation_model,
                                                                          gen_dec_rocksample.agent.transition_model)
                            gen_dec_rocksample.agent.set_belief(new_belief)
                            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                            agent2_obs = get_str_obs(observation_after_execute.quality)
                            agent2_window.append((agent2_step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                            agent2_step += 1
                        else:
                            agent2_didnt_do_idle = False
                            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                            agent2_step+=1
                            #agent 2 end turn --------------------------------------------------------------
                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'
                step+=1
            trace_i+=1
            print(f'success: {terminal_state_flag} in  {step} steps')
            print(agent1_window)
            print(agent2_window)
            print()
            add_results_to_table(trace_i,init_state,agent1_window,agent2_window,terminal_state_flag,step,agents_results)
    return agents_results

def teacher_correction(agent1_window,agent2_window,belief_state,curr_real_state):
    check_point_in_trace = dict()
    check_point_in_trace['agent1_window'] = agent1_window
    check_point_in_trace['agent2_window'] = agent2_window
    check_point_in_trace['belief_state'] = belief_state
    check_point_in_trace['state'] = curr_real_state
    return check_point_in_trace

def execute_for_dec_rock_loadpolicy_tranformer_sync_policyrefine(n,k,bound,rock_loc_param,agents_init_loc,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path,sync_list):
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
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRockEli{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model,_ = load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model,_= load_model(agent2_model_path)
    trace_i=0
    agents_results=dict()
    traces_to_correct = dict()
    index_for_traces_to_correct = 0
    for i in range(0,1):
        for init_state in true_init_bf:
            curr_trace_to_correct=[]
            step = 0
            agent1_step = 0
            agent2_step = 0
            terminal_state_flag = False
            print(f'trace number: {trace_i} init state is : {init_state}')
            gen_dec_rocksample = GenDecRockSampleProblem(n, k, bound, init_state, rocks_locs,
                                                         pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(gen_dec_rocksample.agent)
            reward = gen_dec_rocksample.env.state_transition(action, execute=True)
            observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                          action, observation,
                                                          gen_dec_rocksample.agent.observation_model,
                                                          gen_dec_rocksample.agent.transition_model)
            gen_dec_rocksample.agent.set_belief(new_belief)
            terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'

            #building agent windows
            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
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
                        # print(agent1_emb_seq)
                        agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                        # print(agent1_padded_seq)
                        agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                        agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                        agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                        # print(f"Predicted action: {agent1_predicted_action} which is {agent1_predicted_action_name}")

                        if agent1_predicted_action_name != 'idle':
                            action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name, 1)
                            action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                            action_from_teacher = policy.plan(gen_dec_rocksample.agent)
                            if not checkIfCheckAction(action_to_execute) and action_from_teacher!= action_to_execute:
                                temp_res=teacher_correction(agent1_window,agent2_window,gen_dec_rocksample.agent.cur_belief,gen_dec_rocksample.env.state)
                                curr_trace_to_correct.append(temp_res)

                            reward_after_execute1 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env,
                                                                                            execute=True)
                            observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(
                                gen_dec_rocksample.env.state,
                                action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                          action_to_execute_for_env,
                                                                          observation_after_execute,
                                                                          gen_dec_rocksample.agent.observation_model,
                                                                          gen_dec_rocksample.agent.transition_model)
                            gen_dec_rocksample.agent.set_belief(new_belief)
                            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                            agent1_obs = get_str_obs(observation_after_execute.quality)
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step+=1

                        else:
                            agent1_didnt_do_idle =False
                            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                            agent1_obs = get_str_obs('None')
                            agent1_window.append((agent1_step, agent1_loc, agent1_predicted_action_name, agent1_obs))
                            agent1_step += 1


                else: #agent2_turn
                    agent1_didnt_do_idle = True
                    if agent2_didnt_do_idle == True:
                        #agent 2 turn -----------------------------------------------------------------------------------------------------------
                        agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'], agent2_dict['action_set'], agent2_dict['observation_set'])
                        #print(agent2_emb_seq)
                        agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                        #print(agent2_padded_seq)
                        agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                        agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                        agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                        #print(f"Predicted action: {agent2_predicted_action} which is {agent2_predicted_action_name}")
                        if agent2_predicted_action_name != 'idle':
                            action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name,2)
                            action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                            action_from_teacher = policy.plan(gen_dec_rocksample.agent)
                            if not checkIfCheckAction(action_to_execute) and action_from_teacher!= action_to_execute:
                                temp_res=teacher_correction(agent1_window,agent2_window,gen_dec_rocksample.agent.cur_belief,gen_dec_rocksample.env.state)
                                curr_trace_to_correct.append(temp_res)
                            reward_after_execute2 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env, execute=True)
                            observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                            action_to_execute_for_env)
                            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                          action_to_execute_for_env, observation_after_execute,
                                                                          gen_dec_rocksample.agent.observation_model,
                                                                          gen_dec_rocksample.agent.transition_model)
                            gen_dec_rocksample.agent.set_belief(new_belief)
                            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                            agent2_obs = get_str_obs(observation_after_execute.quality)
                            agent2_window.append((agent2_step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                            agent2_step += 1
                        else:
                            agent2_didnt_do_idle = False
                            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                            agent2_obs = get_str_obs('None')
                            agent2_window.append((agent2_step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                            agent2_step+=1
                            #agent 2 end turn --------------------------------------------------------------
                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'
                step+=1
            trace_i+=1
                        
            if terminal_state_flag is False:
                traces_to_correct[index_for_traces_to_correct] = curr_trace_to_correct
                index_for_traces_to_correct+=1
            print(f'success: {terminal_state_flag} in  {step} steps')
            print(agent1_window)
            print(agent2_window)
            print()
            add_results_to_table(trace_i,init_state,agent1_window,agent2_window,terminal_state_flag,step,agents_results)
    return traces_to_correct

def genrate_traces_for_retrain(traces_for_correction,agent1_path,agent2_path,team_path,n,k,bound,,rocks_locs,agents_init_loc):
    start_time = time.time()
    team, agent1, agent2 = traces_for_retrain(traces_for_correction,n,k,bound,,rocks_locs,agents_init_loc)
    end_time = time.time()
    time_to_make_traces = end_time - start_time
    print(f'took {time_to_make_traces} sec to produce traces')
    save_dict(agent1_path,agent1)
    save_dict(agent2_path,agent2)
    save_dict(team_path,team)
    print('done saving traces')
    return team


def add_results_to_table(trace_number,init_state,agent1_window,agent2_window,terminal_state_flag,step,agents_results):
    agents_results[trace_number]=dict()
    agents_results[trace_number]['init_state'] = init_state
    agents_results[trace_number]['reach_goal']= terminal_state_flag
    steps =0
    samples = ['no'] * len(init_state.rocktypes)
    check_count = 0
    move_count =0
    reward =1500
    for i in range(len(agent1_window)):
        if agent1_window[i][2] != 'idle':
            steps+=1
        if agent1_window[i][2] in {'left','right','up','down'}:
            move_count+=1
        if agent1_window[i][2] in {'check1','check2','check3','check4'}:
            check_count+=1
        if agent1_window[i][2] == 'sample1':
            samples[0] ='yes'
        if agent1_window[i][2] == 'sample2':
            samples[1] ='yes'
        if agent1_window[i][2] == 'sample3':
            samples[2] ='yes'
        if agent1_window[i][2] == 'sample4':
            samples[3] ='yes'
    for i in range(len(agent2_window)):
        if agent2_window[i][2] != 'idle':
            steps+=1
        if agent2_window[i][2] in {'left','right','up','down'}:
            move_count+=1
        if agent2_window[i][2] in {'check1','check2','check3','check4'}:
            check_count+=1
        if agent2_window[i][2] == 'sample1':
            samples[0] ='yes'
        if agent2_window[i][2] == 'sample2':
            samples[1] ='yes'
        if agent2_window[i][2] == 'sample3':
            samples[2] ='yes'
        if agent2_window[i][2] == 'sample4':
            samples[3] ='yes'

    for i in range(len(samples)):
        if init_state.rocktypes[i] == 'B' and samples[i] =='yes':
            reward-=500
        if init_state.rocktypes[i] == 'G' and samples[i] =='no':
            reward-=380
    reward -= move_count*5
    reward -= check_count

    agents_results[trace_number]['steps'] = steps
    agents_results[trace_number]['samples'] = samples
    agents_results[trace_number]['move_count'] = move_count
    agents_results[trace_number]['check_count'] = check_count
    agents_results[trace_number]['reward'] = reward











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
def execute_for_dec_rock_loadpolicy_tranformer_policy_refine_ver1(n,k,bound,rock_loc_param,agents_init_loc,agent1_model_path,agent1_essential_path,agent2_model_path,agent2_essential_path):
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
    policy_path = "%s.policy" % f'decRock_pomdp_artifects_refine/ShrinkedDecRock{n}x{k}x{number_of_rocks}'
    all_states = list(gen_dec_rocksample.agent.all_states)
    all_actions = list(gen_dec_rocksample.agent.all_actions)
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)
    #init agents-------------------------
    agent1_dict = load_and_preprocess(agent1_essential_path)
    agent1_model, _= load_model(agent1_model_path)
    agent2_dict = load_and_preprocess(agent2_essential_path)
    agent2_model, _ = load_model(agent2_model_path)
    trace_i=0
    bad_traces = dict()
    for i in range(0,1):
        for init_state in true_init_bf:
            step = 0
            terminal_state_flag = False
            print(f'trace number: {trace_i} init state is : {init_state}')
            gen_dec_rocksample = GenDecRockSampleProblem(n, k, bound, init_state, rocks_locs,
                                                         pomdp_py.Histogram(init_bf))
            #first action sarsop take
            action = policy.plan(gen_dec_rocksample.agent)
            reward = gen_dec_rocksample.env.state_transition(action, execute=True)
            observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state, action)
            new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                          action, observation,
                                                          gen_dec_rocksample.agent.observation_model,
                                                          gen_dec_rocksample.agent.transition_model)
            gen_dec_rocksample.agent.set_belief(new_belief)
            terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'

            #building agent windows
            agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
            agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
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
            agent1_window = [(step,agent1_loc,agent1_action,agent1_obs)]
            agent2_window = [(step,agent2_loc,agent2_action,agent2_obs)]
            step += 1
            while step<20 and terminal_state_flag == False:
                #agent 1 turn -----------------------
                agent1_emb_seq = embedd_seq(agent1_window, agent1_dict['locations'], agent1_dict['action_set'], agent1_dict['observation_set'])
                #print(agent1_emb_seq)
                agent1_padded_seq = pad_seq(agent1_emb_seq, agent1_dict['max_len_in_train'])
                #print(agent1_padded_seq)
                agent1_input_sequence = torch.tensor(agent1_padded_seq, dtype=torch.float32)
                agent1_predicted_action = predict_next_action(agent1_model, agent1_input_sequence)
                agent1_predicted_action_name = agent1_dict['rev_action_set'][agent1_predicted_action]
                #print(f"Predicted action: {agent1_predicted_action} which is {agent1_predicted_action_name}")
                if agent1_predicted_action_name != 'idle':
                    action_to_execute = get_team_action_from_single_action(agent1_predicted_action_name,1)
                    action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                    reward_after_execute1 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env, execute=True)
                    observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_execute_for_env)
                    new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_execute_for_env, observation_after_execute,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                    gen_dec_rocksample.agent.set_belief(new_belief)
                    agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                    agent1_obs = get_str_obs(observation_after_execute.quality)
                    agent1_window.append((step,agent1_loc,agent1_predicted_action_name,agent1_obs))
                else:
                    agent1_loc = (gen_dec_rocksample.env.cur_state.position[0], gen_dec_rocksample.env.cur_state.position[1])
                    agent1_obs = get_str_obs('None')
                    agent1_window.append((step,agent1_loc,agent1_predicted_action_name,agent1_obs))

                #agent1 end turn
                #agent 2 turn -----------------------------------------------------------------------------------------------------------
                agent2_emb_seq = embedd_seq(agent2_window, agent2_dict['locations'], agent2_dict['action_set'], agent2_dict['observation_set'])
                #print(agent2_emb_seq)
                agent2_padded_seq = pad_seq(agent2_emb_seq, agent2_dict['max_len_in_train'])
                #print(agent2_padded_seq)
                agent2_input_sequence = torch.tensor(agent2_padded_seq, dtype=torch.float32)
                agent2_predicted_action = predict_next_action(agent2_model, agent2_input_sequence)
                agent2_predicted_action_name = agent2_dict['rev_action_set'][agent2_predicted_action]
                #print(f"Predicted action: {agent2_predicted_action} which is {agent2_predicted_action_name}")
                if agent2_predicted_action_name != 'idle':
                    action_to_execute = get_team_action_from_single_action(agent2_predicted_action_name,2)
                    action_to_execute_for_env = GenDecRockSampleAction(action_to_execute)
                    reward_after_execute2 = gen_dec_rocksample.env.state_transition(action_to_execute_for_env, execute=True)
                    observation_after_execute = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,
                                                                                    action_to_execute_for_env)
                    new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                                  action_to_execute_for_env, observation_after_execute,
                                                                  gen_dec_rocksample.agent.observation_model,
                                                                  gen_dec_rocksample.agent.transition_model)
                    gen_dec_rocksample.agent.set_belief(new_belief)
                    agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                    agent2_obs = get_str_obs(observation_after_execute.quality)
                    agent2_window.append((step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                else:
                    agent2_loc = (gen_dec_rocksample.env.cur_state.position[2], gen_dec_rocksample.env.cur_state.position[3])
                    agent2_obs = get_str_obs('None')
                    agent2_window.append((step,agent2_loc,agent2_predicted_action_name,agent2_obs))
                #agent 2 end turn --------------------------------------------------------------
                terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'
                step+=1

            print(f'success: {terminal_state_flag} in  {step} steps')
            print(agent1_window)
            print(agent2_window)
            print()
            if terminal_state_flag ==False:
                bad_curr =dict()
                bad_curr['init_state'] = init_state
                bad_curr['agent1'] = agent1_window
                bad_curr['agent2'] = agent2_window
                bad_traces[trace_i] = bad_curr

            trace_i += 1
    return bad_traces

def correct_traces_v1(n,k,bound,rock_loc_param,agents_init_loc,bad_traces):
    team_traces = dict()
    agent1_traces = dict()
    agent2_traces = dict()
    rocks_locs = rock_loc_param
    number_of_rocks = len(rocks_locs)
    rock_tup_exmp = tuple('G') * number_of_rocks
    init_state = GenDecRockSampleState(agents_init_loc, rock_tup_exmp)
    all_states = gen_dec_get_all_states(n, k, bound, number_of_rocks)
    init_bf = dict()
    for state in all_states:
        init_bf[state] = 0

    lst = ['G', 'B']
    rocks = (list(itertools.product(lst, repeat=number_of_rocks)))
    true_init_bf = []
    for rock in rocks:
        tup_rock = tuple(rock)
        true_init_bf.append(GenDecRockSampleState(agents_init_loc, tup_rock))
    init_belief_prob = 1 / len(true_init_bf)
    for bstate in true_init_bf:
        init_bf[bstate] = init_belief_prob
    gen_dec_rocksample = GenDecRockSampleProblem(n, k, bound, init_state, rocks_locs,
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
    trace_i = 0
    for index,tr in bad_traces.items():
        terminal_state_flag = False
        init_state = tr['init_state']
        gen_dec_rocksample = GenDecRockSampleProblem(n, k, bound, init_state, rocks_locs,
                                                     pomdp_py.Histogram(init_bf))
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
        step = 0
        insdie_trace_index = 0
        teacher_correcting = False
        while step < 20 and terminal_state_flag == False:
            if teacher_correcting == True:
                action = policy.plan(gen_dec_rocksample.agent)
                reward = gen_dec_rocksample.env.state_transition(action,execute= True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state,action)
                add_to_trace(gen_dec_rocksample,action,reward,observation,team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)

                gen_dec_rocksample.agent.set_belief(new_belief)
            else:
                action_to_execute = None
                teacher_action = policy.plan(gen_dec_rocksample.agent)
                action_from_trace = 'None'
                action_from_trace_agent1 = tr['agent1'][insdie_trace_index][2]
                action_from_trace_agent2 = tr['agent2'][insdie_trace_index][2]
                if action_from_trace_agent1!= 'idle' and action_from_trace_agent2!= 'idle':
                    print('bugggggg two actions no idle')
                elif action_from_trace_agent1 != 'idle':
                    action_from_trace = get_team_action_from_single_action(action_from_trace_agent1,1)
                else:
                    action_from_trace = get_team_action_from_single_action(action_from_trace_agent2, 2)

                if checkIfCheckAction(action_from_trace) or action_from_trace == teacher_action.name :
                    action_to_execute = GenDecRockSampleAction(action_from_trace)
                else:
                    action_to_execute = teacher_action
                    teacher_correcting = True

                reward = gen_dec_rocksample.env.state_transition(action_to_execute, execute=True)
                observation = gen_dec_rocksample.agent.observation_model.sample(gen_dec_rocksample.env.state, action_to_execute)
                add_to_trace(gen_dec_rocksample, action_to_execute, reward, observation, team_trace, agent1_trace, agent2_trace)
                new_belief = pomdp_py.update_histogram_belief(gen_dec_rocksample.agent.cur_belief,
                                                              action_to_execute, observation,
                                                              gen_dec_rocksample.agent.observation_model,
                                                              gen_dec_rocksample.agent.transition_model)

                gen_dec_rocksample.agent.set_belief(new_belief)
            terminal_state_flag = gen_dec_rocksample.env.cur_state.position[0] == 'T'
            step += 1
            insdie_trace_index+=1
        team_traces[trace_i] = team_trace
        agent1_traces[trace_i] = agent1_trace
        agent2_traces[trace_i] = agent2_trace
        trace_i += 1
    return  team_traces,agent1_traces,agent2_traces


def get_action_traces_for_sync(agent_dict):
    agent_actions_sync_list=[]
    for num,trace_dict in agent_dict.items():
        tr = []
        for i in range(len(trace_dict['actions'])):
            if which_agent_operate_now(trace_dict['actions'][i]) ==1:
                tr.append('agent1')
            else:
                tr.append('agent2')
        agent_actions_sync_list.append(tr)
    return agent_actions_sync_list


def get_maximal_intervals(trace_list):
    ret_list = []
    graph = dict()
    graph[0]=['agent1',0]
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
        ret_list.extend([graph[i][0]] * (graph[i][1]+1) )# plus 1 is for the idle i keep
    return ret_list












def save_agents_dicts(problem_name,date,team,agent1,agent2):
    with open(f'traces/{problem_name}_{date}_team.pickle', 'wb') as handle:
        pickle.dump(team, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'traces/{problem_name}_{date}_agent1.pickle', 'wb') as handle:
        pickle.dump(agent1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'traces/{problem_name}_{date}_agent2.pickle', 'wb') as handle:
        pickle.dump(agent2, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    print('PyCharm hello')
    date_for_traces='06_08_2023'
    date= '07_08_2023'
    n=4
    k = 5
    rnum = 4
    """agent1_path = 'traces/dec_rock4x3x3_22_07_2023_agent1.pickle'
    agent2_path = 'traces/dec_rock4x3x3_22_07_2023_agent2.pickle'
    agent1_bad_trace_path = 'traces/dec_rock_bad_traces4x3x3_05_08_2023_agent1.pickle'
    agent2_bad_trace_path = 'traces/dec_rock_bad_traces4x3x3_05_08_2023_agent2.pickle'
    agent1_save_model_path = 'transformer_saves/dec_rock4x3x3_05_08_2023_agent1.pth'
    retrained_agent1_save_model_path = 'transformer_saves/dec_rock4x3x3_05_08_2023_agent1_retrain.pth'
    agent1_save_essential_path = 'transformer_saves/dec_rock4x3x3_05_08_2023_agent1_essential.pickle'
    agent2_save_model_path = 'transformer_saves/dec_rock4x3x3_05_08_2023_agent2.pth'
    retrained_agent2_save_model_path = 'transformer_saves/dec_rock4x3x3_05_08_2023_agent2_retrain.pth'
    agent2_save_essential_path = 'transformer_saves/dec_rock4x3x3_05_08_2023_agent2_essential.pickle'"""
    team_path = f'traces/dec_rock{n}x{k}x{rnum}_{date_for_traces}_team.pickle'
    agent1_path = f'traces/dec_rock{n}x{k}x{rnum}_{date_for_traces}_agent1.pickle'
    agent2_path = f'traces/dec_rock{n}x{k}x{rnum}_{date_for_traces}_agent2.pickle'
    agent1_bad_trace_path = f'traces/dec_rock_bad_traces{n}x{k}x{rnum}_{date}_agent1.pickle'
    agent2_bad_trace_path = f'traces/dec_rock_bad_traces{n}x{k}x{rnum}_{date}_agent2.pickle'
    agent1_save_model_path = f'transformer_saves/dec_rock{n}x{k}x{rnum}_{date}_agent1.pth'
    retrained_agent1_save_model_path = f'transformer_saves/dec_rock{n}x{k}x{rnum}_{date}_agent1_retrain.pth'
    agent1_save_essential_path = f'transformer_saves/dec_rock{n}x{k}x{rnum}_{date}_agent1_essential.pickle'
    agent2_save_model_path = f'transformer_saves/dec_rock{n}x{k}x{rnum}_{date}_agent2.pth'
    retrained_agent2_save_model_path = f'transformer_saves/dec_rock{n}x{k}x{rnum}_{date}agent2_retrain.pth'
    agent2_save_essential_path = f'transformer_saves/dec_rock{n}x{k}x{rnum}_{date}_agent2_essential.pickle'
    team_sync_path = f'results/dec_rock_sync{n}x{k}x{rnum}_{date}.pickle'
    phase = 8
    if phase ==1:# train agents for the first time

        agent1_dict = load_and_preprocess(agent1_path)
        agent2_dict = load_and_preprocess(agent2_path)
        agent1_traces = get_traces(agent1_dict)
        agent2_traces = get_traces(agent2_dict)
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)

        train_agent1(agent1_dict,agent1_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path)

        train_agent2(agent2_dict,agent2_traces,max_len_agent2,agent2_save_model_path,agent2_save_essential_path)
    if phase ==2: #check trained agents
        n = 4
        k = 3
        bound = 1
        rocks_locs = [(0, 2), (1, 3), (1, 0),(2,1)]
        agents_init_loc = (0, 3, 2, 0)
        execute_for_dec_rock_loadpolicy_tranformer(n,k,bound,rocks_locs,agents_init_loc,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path)
    if phase == 3: #looking for bad traces to retrain and let team solve them
        n = 4
        k = 3
        bound = 1
        rocks_locs = [(0, 1), (1, 0), (1, 2)]
        agents_init_loc = (0, 0, 2, 0)
        start_time_gen_bad_traces = time.time()
        bad_traces = execute_for_dec_rock_loadpolicy_tranformer_policy_refine_ver1(n,k,bound,rocks_locs,agents_init_loc,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path)
        if len(bad_traces) > 0:
            team,agent1,agent2 = correct_traces_v1(n,k,bound,rocks_locs,agents_init_loc,bad_traces)
            save_agents_dicts(f'dec_rock_bad_traces{n}x{k}x{len(rocks_locs)}', '04_08_2023', team, agent1, agent2)
            print(agent1)
            print(agent2)
        endtime_gen_bad_traces = time.time()
        time_to_make_bad_traces = endtime_gen_bad_traces - start_time_gen_bad_traces
        print(f'took {time_to_make_bad_traces} sec to produce traces')
    if phase == 4: #retraining on bad traces
        agent1_bad_dict = load_and_preprocess(agent1_bad_trace_path)
        agent2_bad_dict = load_and_preprocess(agent2_bad_trace_path)
        agent1_bad_traces = get_traces(agent1_bad_dict)
        agent2_bad_traces = get_traces(agent2_bad_dict)

        agent1_dict = load_and_preprocess(agent1_path)
        agent2_dict = load_and_preprocess(agent2_path)
        agent1_traces = get_traces(agent1_dict)
        agent2_traces = get_traces(agent2_dict)
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)
        agent1_model = retrain_agent(agent1_bad_dict,agent1_bad_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path,retrained_agent1_save_model_path)
        agent2_model = retrain_agent(agent2_bad_dict, agent2_bad_traces, max_len_agent2, agent2_save_model_path,
                                     agent2_save_essential_path, retrained_agent2_save_model_path)
    if phase == 5: #check after retrain
        n = 4
        k = 3
        bound = 1
        rocks_locs = [(0, 1), (1, 0), (1, 2)]
        agents_init_loc = (0, 0, 2, 0)
        execute_for_dec_rock_loadpolicy_tranformer(n,k,bound,rocks_locs,agents_init_loc,retrained_agent1_save_model_path,agent1_save_essential_path,retrained_agent2_save_model_path,agent2_save_essential_path)
    if phase == 6: #retraining on bad traces but choose the traces myself
        agent1_dict = load_and_preprocess(agent1_path)
        agent2_dict = load_and_preprocess(agent2_path)
        agent1_traces = get_traces(agent1_dict)
        agent1_bad_traces = [agent1_traces[5]]
        agent2_traces = get_traces(agent2_dict)
        agent2_bad_traces = [agent2_traces[5]]
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)
        agent1_model = retrain_agent(agent1_dict,agent1_bad_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path,retrained_agent1_save_model_path)
        agent2_model = retrain_agent(agent2_dict, agent2_bad_traces, max_len_agent2, agent2_save_model_path,
                                     agent2_save_essential_path, retrained_agent2_save_model_path)
    if phase ==7: #sync agents
        team_dict = load_and_preprocess(team_path)
        traces= get_action_traces_for_sync(team_dict)
        result = get_maximal_intervals(traces)
        dict_result = {'sync_list': result}
        save_dict(team_sync_path, dict_result)
        agent1_dict = load_and_preprocess(agent1_path)
        agent2_dict = load_and_preprocess(agent2_path)
        agent1_traces = get_traces_only_one_idle(agent1_dict)
        agent2_traces = get_traces_only_one_idle(agent2_dict)
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)

        train_agent1(agent1_dict,agent1_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path)

        train_agent2(agent2_dict,agent2_traces,max_len_agent2,agent2_save_model_path,agent2_save_essential_path)
    if phase ==8: #check trained agents synced
        result_from_sync = load_and_preprocess(team_sync_path)
        result_from_sync = result_from_sync['sync_list']
        n = 4
        k = 5
        bound = 2
        rocks_locs = [(0, 2), (2, 3), (2, 0),(44,1)]
        agents_init_loc = (0, 3, 4, 0)
        start_time = time.time()
        results=execute_for_dec_rock_loadpolicy_tranformer_sync(n,k,bound,rocks_locs,agents_init_loc,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
        end_time = time.time()
        time_to_detect_bad_traces = end_time - start_time
        print(f'took {time_to_detect_bad_traces} sec to produce traces')
        tries = len(results.keys())
        sum_reward = 0
        sum_steps= 0
        succ_rate = 0
        for i,trace_res in results.items():
            print(trace_res)
            sum_steps+=trace_res['steps']
            sum_reward+= trace_res['reward']
            if trace_res['reach_goal'] ==True:
                succ_rate+=1
        print(sum_steps/tries)
        print(sum_reward / tries)
        print(succ_rate / tries)



    if phase == 9:
        team_dict = load_and_preprocess(team_path)
        traces= get_action_traces_for_sync(team_dict)
        result = get_maximal_intervals(traces)
        dict_result = {'sync_list': result}
        save_dict(team_sync_path, dict_result)
        agent1_dict = load_and_preprocess(agent1_path)
        agent2_dict = load_and_preprocess(agent2_path)
        agent1_traces = get_traces_only_one_idle(agent1_dict)
        agent2_traces = get_traces_only_one_idle(agent2_dict)
        max_len_agent1 = get_max_len(agent1_traces)
        max_len_agent2 = get_max_len(agent2_traces)

        train_agent1(agent1_dict,agent1_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path)

        train_agent2(agent2_dict,agent2_traces,max_len_agent2,agent2_save_model_path,agent2_save_essential_path)
        #---------------------------#

        result_from_sync = load_and_preprocess(team_sync_path)
        result_from_sync = result_from_sync['sync_list']
        n = 4
        k = 5
        bound = 2
        rocks_locs = [(0, 2), (2, 3), (2, 0),(44,1)]
        agents_init_loc = (0, 3, 4, 0)
        start_time = time.time()
        traces_for_corrections=execute_for_dec_rock_loadpolicy_tranformer_sync_policyrefine(n,k,bound,rocks_locs,agents_init_loc,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
        team_traces=genrate_traces_for_retrain(traces_for_correction,agent1_retrain_traces_path,agent2_retrain_traces_path,team_retrain_path,n,k,bound,,rocks_locs,agents_init_loc)
        max_refine_steps = 4
        i = 0
        while(i<max_refine_steps and len(traces_for_correction>0) ):
            agent1_bad_dict = load_and_preprocess(agent1_bad_trace_path)
            agent2_bad_dict = load_and_preprocess(agent2_bad_trace_path)
            agent1_bad_traces = get_traces(agent1_bad_dict)
            agent2_bad_traces = get_traces(agent2_bad_dict)

            agent1_dict = load_and_preprocess(agent1_path)
            agent2_dict = load_and_preprocess(agent2_path)
            agent1_traces = get_traces(agent1_dict)
            agent2_traces = get_traces(agent2_dict)
            max_len_agent1 = get_max_len(agent1_traces)
            max_len_agent2 = get_max_len(agent2_traces)
            agent1_model = retrain_agent(agent1_bad_dict,agent1_bad_traces,max_len_agent1,agent1_save_model_path,agent1_save_essential_path,retrained_agent1_save_model_path)
            agent2_model = retrain_agent(agent2_bad_dict, agent2_bad_traces, max_len_agent2, agent2_save_model_path,
                                        agent2_save_essential_path, retrained_agent2_save_model_path)



            #-------------#
            result_from_sync = load_and_preprocess(team_sync_path)
            result_from_sync = result_from_sync['sync_list']
            start_time = time.time()
            # results=execute_for_box_push_loadpolicy_tranformer_sync(init_bf,true_init_bf,policy,n,k,box_type,prob_for_push,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
            traces_for_corrections=execute_for_dec_rock_loadpolicy_tranformer_sync_policyrefine(n,k,bound,rocks_locs,agents_init_loc,agent1_save_model_path,agent1_save_essential_path,agent2_save_model_path,agent2_save_essential_path,result_from_sync)
            team_traces=genrate_traces_for_retrain(traces_for_correction,agent1_retrain_traces_path,agent2_retrain_traces_path,team_retrain_path,n,k,bound,,rocks_locs,agents_init_loc)
            i+=1
        







    #code for checking time
    """start_time = time.time()   
    end_time = time.time()
    time_to_make_traces = end_time - start_time
    print(f'took {time_to_make_traces} sec to produce traces')"""

