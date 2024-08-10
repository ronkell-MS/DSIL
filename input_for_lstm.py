import pomdp_py
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pomdp_problems.tiger.tiger_problem import TigerProblem , TigerState
from pomdp_problems.dec_tiger.dec_tiger_problem import DecTigerProblem, DecTigerState, DecTigerAction
from pomdp_problems.rocksample.rocksample_problem import RockSampleProblem, State, init_particles_belief
from pomdp_problems.dec_rocksample.dec_rocksample_problem import DecRockSampleProblem, DecRockSampleState, DecRockSampleAction
from pomdp_problems.small_dec_rock.small_dec_rock_problem import SmallDecRockSampleProblem, SmallDecRockSampleState, SmallDecRockSampleAction, SmallDecRockSampleObservation, checkIfCheckAction
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import numpy as np
import tensorflow as tf
import time
def load_and_preprocess(pickle_path):
    with open(pickle_path, 'rb') as handle:
        agentdict = pickle.load(handle)
    return agentdict
def create_vocab(loc_bounds,actions,observations):
    ret_dict = dict()
    reveres_dict = dict()
    count = 1
    ret_dict[(-1,-1,-1,-1)] = 0
    reveres_dict[0]=(-1,-1,-1,-1)
    for i in range(loc_bounds[0],loc_bounds[1]):
        for j in range(loc_bounds[2],loc_bounds[3]):
            for act in actions:
                if act in ['check1','check2','check3']:
                    for obs in ['G','B']:
                        ret_dict[(i, j, obs, act)] = count
                        reveres_dict[count] = (i, j, obs, act)
                        count += 1
                else:
                    ret_dict[(i, j, 'None', act)] = count
                    reveres_dict[count] = (i, j, 'None', act)
                    count += 1

    for act in actions:
        if act in ['check1', 'check2','check3']:
            for obs in ['G', 'B']:
                ret_dict[('T', 'T', obs, act)] = count
                reveres_dict[count] = ('T', 'T', obs, act)
                count += 1
        else:
            ret_dict[('T', 'T', 'None', act)] = count
            reveres_dict[count] = ('T', 'T', 'None', act)
            count += 1

    """for i in range(loc_bounds[0],loc_bounds[1]):
        for j in range(loc_bounds[2],loc_bounds[3]):
            for obs in observations:
                for act in actions:
                    ret_dict[(i, j, obs, act)] = count
                    reveres_dict[count] = (i, j, obs, act)
                    count += 1"""

    return ret_dict,reveres_dict

def create_vocab_for_time(loc_bounds,actions,observations,max_len):
    ret_dict = dict()
    reveres_dict = dict()
    count = 1
    ret_dict[(-1,-1,-1,-1,-1)] = 0
    reveres_dict[0]=(-1,-1,-1,-1,-1)
    for t in range(0,max_len):
        for i in range(loc_bounds[0],loc_bounds[1]):
            for j in range(loc_bounds[2],loc_bounds[3]):
                for act in actions:
                    if act in ['check1','check2']:
                        for obs in ['G','B']:
                            ret_dict[(t,i, j, obs, act)] = count
                            reveres_dict[count] = (t,i, j, obs, act)
                            count += 1
                    else:
                        ret_dict[(t,i, j, 'None', act)] = count
                        reveres_dict[count] = (t,i, j, 'None', act)
                        count += 1

        for act in actions:
            if act in ['check1', 'check2']:
                for obs in ['G', 'B']:
                    ret_dict[(t,'T', 'T', obs, act)] = count
                    reveres_dict[count] = (t,'T', 'T', obs, act)
                    count += 1
            else:
                ret_dict[(t,'T', 'T', 'None', act)] = count
                reveres_dict[count] = (t,'T', 'T', 'None', act)
                count += 1

    """for i in range(loc_bounds[0],loc_bounds[1]):
        for j in range(loc_bounds[2],loc_bounds[3]):
            for obs in observations:
                for act in actions:
                    ret_dict[(i, j, obs, act)] = count
                    reveres_dict[count] = (i, j, obs, act)
                    count += 1"""

    return ret_dict,reveres_dict
def create_actions__vocab(actions):
    ret_dict = dict()
    reversed_dict = dict()
    count = 1
    ret_dict[-1] = 0
    reversed_dict[0]=-1
    for act in actions:
        ret_dict[act]=count
        reversed_dict[count]=act
        count+=1
    return ret_dict,reversed_dict

def get_max_len_seq(agent_dict):
    max_len = 0
    for index, trace in agent_dict.items():
        actions = trace['actions']
        max_len = max(max_len,len(actions))
    return max_len

def building_seq_data(agent_dict,len_of_seq=3,vocab_dict=None,time_flag=False):
    sequences=[]
    sequences_with_index=[]
    tokenized_seq=[]
    for index,trace in agent_dict.items():
        actions=trace['actions']
        observations=trace['observations']
        locations = trace['states']
        time_step=list(range(len(actions)))
        padding=[-1]*(len_of_seq-1)
        location_padding = [(-1,-1)] * (len_of_seq - 1)
        padded_actions=padding+actions
        padded_observations=padding+observations
        padded_timestep=padding+time_step
        padded_location = location_padding+locations
        for i in range(len_of_seq,len(padded_actions)+1):
            action_seq=padded_actions[i-len_of_seq:i]
            obs_seq=padded_observations[i-len_of_seq:i]
            time_step_seq=padded_timestep[i-len_of_seq:i]
            location_seq = padded_location[i-len_of_seq:i]
            indexes=list(range(i-len_of_seq,i))
            location_obs_action_seq=[]
            tokenize_location_obs_action_seq = []
            timestep_locations_obs_action_seq=[]
            for j in range(len(action_seq)):
                location_obs_action_seq.append((location_seq[j][0],location_seq[j][1],obs_seq[j],action_seq[j]))
                timestep_locations_obs_action_seq.append((time_step_seq[j],location_seq[j][0],location_seq[j][1], obs_seq[j], action_seq[j]))
                if vocab_dict != None:
                    if time_flag == False:
                        tokenize_location_obs_action_seq.append(vocab_dict[(location_seq[j][0],location_seq[j][1],obs_seq[j],action_seq[j])])
                    else:
                        tokenize_location_obs_action_seq.append(vocab_dict[(time_step_seq[j],location_seq[j][0], location_seq[j][1], obs_seq[j], action_seq[j])])

            sequences.append(location_obs_action_seq)
            sequences_with_index.append(timestep_locations_obs_action_seq)
            tokenized_seq.append(tokenize_location_obs_action_seq)
    return sequences,sequences_with_index,tokenized_seq

def train_split_label_action(sequences):
    x_train=[]
    y_train=[]
    for i in range(len(sequences)):
        x_train.append(sequences[i][:-1])
        y_train.append(sequences[i][-1][-1])
    return x_train,y_train
def train_split_label_action_from_tokenize_and_not_tokenize(sequences,untokenize_seq,actions_vocab):
    x_train=[]
    y_train=[]
    for i in range(len(sequences)):
        x_train.append(sequences[i][:-1])
        y_train.append(actions_vocab[untokenize_seq[i][-1][-1]])
    return x_train,y_train

def train_split_label(sequences):
    x_train=[]
    y_train=[]
    for i in range(len(sequences)):
        x_train.append(sequences[i][:-1])
        y_train.append(sequences[i][-1])
    return x_train,y_train

def buildmoelwithEmbbeding(vocab_size,seq_len):
    model = Sequential()
    model.add(Embedding(vocab_size,vocab_size,input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    # compiling the network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def buildmoelwithEmbbedingWithActionLabel(vocab_size,seq_len,output_vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size,vocab_size,input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_vocab_size, activation='softmax'))
    # compiling the network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def turn_num_to_vec(num,vocab):
    arr=[0]*(len(vocab))
    arr[num] = 1
    return np.array(arr)

def turn_label_to_vec(labels,vocab):
    ret_arr = []
    for i in range(0,len(labels)):
        ret_arr.append(turn_num_to_vec(labels[i],vocab))
    return ret_arr
def train_model(model,x_train,y_train,epochs_num=20):
    history=model.fit(x_train, y_train, epochs=epochs_num, verbose=1)
    return history

def predict(model,x_test):
    preds = model.predict(x_test)
    preds_classes = np.argmax(preds, axis=-1)
    return preds_classes


def run_regular_pipe():
    team_dict = load_and_preprocess('traces/dec_rock_04_02_2023_team.pickle')
    agent1_dict = load_and_preprocess('traces/dec_rock_04_02_2023_agent1.pickle')
    agent2_dict = load_and_preprocess('traces/dec_rock_04_02_2023_agent2.pickle')
    agent1_bounds = [0,2,0,3]
    agent1_actions = ['idle','up','down','left','right','check1','check2','sample1','sample2']
    agent1_obs = ['None','G','B']
    # ------- train model without time step-------#
    full_agent1_vocab,full_agent1_reverse_vocab=create_vocab(agent1_bounds,agent1_actions,agent1_obs)
    max_seq_len = get_max_len_seq(agent1_dict)
    time_flag = False
    agent1_seq,agent1_time_seq,agent1_tokenize_seq = building_seq_data(agent1_dict,max_seq_len,full_agent1_vocab,time_flag)
    train,labels = train_split_label(agent1_seq)
    x_train, y_train = train_split_label(agent1_tokenize_seq)
    y_train_vec = turn_label_to_vec(y_train,full_agent1_vocab)
    agent1_model = buildmoelwithEmbbeding(len(full_agent1_vocab),max_seq_len-1)
    hist = train_model(agent1_model,np.array(x_train),np.array(y_train_vec),epochs_num=100)
    x_test = x_train[0:10]
    preds = agent1_model.predict(np.array(x_test))
    #sort_1 = preds.argsort().tolist()
    #sort_2 = np.argmax(preds[0],axis=-1)
    preds_actual = []
    for x_sample in x_test:
        preds = agent1_model.predict(np.array([x_sample]))
        max_pred = np.argmax(preds,axis=-1)[0]
        preds_actual.append(full_agent1_reverse_vocab[max_pred])
    print(labels)
def run_timed_seq_pipe():
    team_dict = load_and_preprocess('traces/dec_rock_03_02_2023_team.pickle')
    agent1_dict = load_and_preprocess('traces/dec_rock_03_02_2023_agent1.pickle')
    agent2_dict = load_and_preprocess('traces/dec_rock_03_02_2023_agent2.pickle')
    agent1_bounds = [0,2,0,3]
    agent1_actions = ['idle','up','down','left','right','check1','check2','sample1','sample2']
    agent1_obs = ['None','G','B']
    # ------- train model with time step-------#
    max_seq_len = get_max_len_seq(agent1_dict)
    full_agent1_vocab,full_agent1_reverse_vocab=create_vocab_for_time(agent1_bounds,agent1_actions,agent1_obs,max_seq_len)
    time_flag = True
    agent1_seq,agent1_time_seq,agent1_tokenize_seq = building_seq_data(agent1_dict,max_seq_len,full_agent1_vocab,time_flag)
    train,labels = train_split_label(agent1_time_seq)
    x_train, y_train = train_split_label(agent1_tokenize_seq)
    y_train_vec = turn_label_to_vec(y_train,full_agent1_vocab)
    agent1_model = buildmoelwithEmbbeding(len(full_agent1_vocab),max_seq_len-1)
    hist = train_model(agent1_model,np.array(x_train),np.array(y_train_vec),epochs_num=100)
    x_test = x_train[0:10]
    preds = agent1_model.predict(np.array(x_test))
    #sort_1 = preds.argsort().tolist()
    #sort_2 = np.argmax(preds[0],axis=-1)
    preds_actual = []
    for x_sample in x_test:
        preds = agent1_model.predict(np.array([x_sample]))
        max_pred = np.argmax(preds,axis=-1)[0]
        preds_actual.append(full_agent1_reverse_vocab[max_pred])
    print(labels)


def run_pipe_with_action_labels(essential_path,traces_path,agent_bounds):
    """
    run the pipe with actions as labels so the dict output is smaller
    :return:
    """
    #--------------- paths----------
    #save_path = 'dec_rock_essential/11_02_2023/agent1/'
    save_path = essential_path
    model_path = save_path+'model.h5'
    vocab_path = save_path+'vocab.pickle'
    reverse_vocab_path = save_path+'reverse_vocab.pickle'
    actions_path = save_path+'action_vocab.pickle'
    reverse_action_path = save_path+'reverse_action_vocab.pickle'
    max_len_path = save_path+'seq_len.pickle'
    # ----------- end paths ---------------------
    """team_dict = load_and_preprocess('traces/dec_rock_04_02_2023_team.pickle')
    agent1_dict = load_and_preprocess('traces/dec_rock_04_02_2023_agent1.pickle')
    agent2_dict = load_and_preprocess('traces/dec_rock_04_02_2023_agent2.pickle')"""
    team_dict = load_and_preprocess(f'traces/{traces_path}_team.pickle')
    agent1_dict = load_and_preprocess(f'traces/{traces_path}_agent1.pickle')
    agent2_dict = load_and_preprocess(f'traces/{traces_path}_agent2.pickle')
    #agent1_bounds = [0,2,0,3]
    agent1_bounds = agent_bounds
    agent1_actions = ['idle','up','down','left','right','check1','check2','sample1','sample2']
    agent1_obs = ['None','G','B']
    # ------- train model without time step-------#
    full_agent1_vocab,full_agent1_reverse_vocab=create_vocab(agent1_bounds,agent1_actions,agent1_obs)
    action_vocab,reversed_action_vocab = create_actions__vocab(agent1_actions)
    max_seq_len = get_max_len_seq(agent1_dict)
    time_flag = False
    agent1_seq,agent1_time_seq,agent1_tokenize_seq = building_seq_data(agent1_dict,max_seq_len,full_agent1_vocab,time_flag)
    train,labels = train_split_label(agent1_seq)
    x_train,y_train = train_split_label_action_from_tokenize_and_not_tokenize(agent1_tokenize_seq,agent1_seq,action_vocab)
    y_train_vec = turn_label_to_vec(y_train,action_vocab)
    agent1_model = buildmoelwithEmbbedingWithActionLabel(len(full_agent1_vocab),max_seq_len-1,len(action_vocab))
    hist = train_model(agent1_model,np.array(x_train),np.array(y_train_vec),epochs_num=100)
    # ----- save essential ----------------
    with open(vocab_path, 'wb') as handle:
        pickle.dump(full_agent1_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reverse_vocab_path, 'wb') as handle:
        pickle.dump(full_agent1_reverse_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(actions_path, 'wb') as handle:
        pickle.dump(action_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reverse_action_path, 'wb') as handle:
        pickle.dump(reversed_action_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    max_len_dict = {}
    max_len_dict['len_seq'] = max_seq_len
    with open(max_len_path, 'wb') as handle:
        pickle.dump(max_len_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    agent1_model.save(model_path)
    # ---------stop save---------------------

    agent1_model = tf.keras.models.load_model(model_path)
    x_test = x_train[0:10]
    preds = agent1_model.predict(np.array(x_test))
    #sort_1 = preds.argsort().tolist()
    #sort_2 = np.argmax(preds[0],axis=-1)
    preds_actual = []
    for x_sample in x_test:
        preds = agent1_model.predict(np.array([x_sample]))
        max_pred = np.argmax(preds,axis=-1)[0]
        preds_actual.append(reversed_action_vocab[max_pred])
    x_test2 = x_train[48:55]
    preds_actual2= []
    for x_sample in x_test2:
        preds = agent1_model.predict(np.array([x_sample]))
        max_pred = np.argmax(preds,axis=-1)[0]
        preds_actual2.append(reversed_action_vocab[max_pred])
    print(labels)




def run_pipe_with_action_labels_for_agent2(essential_path,traces_path,agent_bounds):
    """
    run the pipe with actions as labels so the dict output is smaller
    :return:
    """
    save_path = essential_path
    model_path = save_path+'model.h5'
    vocab_path = save_path+'vocab.pickle'
    reverse_vocab_path = save_path+'reverse_vocab.pickle'
    actions_path = save_path+'action_vocab.pickle'
    reverse_action_path = save_path+'reverse_action_vocab.pickle'
    max_len_path = save_path+'seq_len.pickle'

    """team_dict = load_and_preprocess('traces/dec_rock_04_02_2023_team.pickle')
    agent2_dict = load_and_preprocess('traces/dec_rock_04_02_2023_agent2.pickle')
    agent2_bounds = [1,3,0,3]"""

    team_dict = load_and_preprocess(f'traces/{traces_path}_team.pickle')
    agent2_dict = load_and_preprocess(f'traces/{traces_path}_agent2.pickle')
    agent2_bounds = agent_bounds

    agent2_actions = ['idle','up','down','left','right','check3','check2','sample3','sample2']
    agent2_obs = ['None','G','B']
    # ------- train model without time step-------#
    full_agent2_vocab,full_agent2_reverse_vocab=create_vocab(agent2_bounds,agent2_actions,agent2_obs)
    action_vocab,reversed_action_vocab = create_actions__vocab(agent2_actions)
    max_seq_len = get_max_len_seq(agent2_dict)
    time_flag = False
    agent2_seq,agent2_time_seq,agent2_tokenize_seq = building_seq_data(agent2_dict,max_seq_len,full_agent2_vocab,time_flag)
    train,labels = train_split_label(agent2_seq)
    x_train,y_train = train_split_label_action_from_tokenize_and_not_tokenize(agent2_tokenize_seq,agent2_seq,action_vocab)
    y_train_vec = turn_label_to_vec(y_train,action_vocab)
    agent2_model = buildmoelwithEmbbedingWithActionLabel(len(full_agent2_vocab),max_seq_len-1,len(action_vocab))
    hist = train_model(agent2_model,np.array(x_train),np.array(y_train_vec),epochs_num=100)
    # ----- save essential ----------------
    with open(vocab_path, 'wb') as handle:
        pickle.dump(full_agent2_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reverse_vocab_path, 'wb') as handle:
        pickle.dump(full_agent2_reverse_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(actions_path, 'wb') as handle:
        pickle.dump(action_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reverse_action_path, 'wb') as handle:
        pickle.dump(reversed_action_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    max_len_dict = {}
    max_len_dict['len_seq']=max_seq_len
    with open(max_len_path, 'wb') as handle:
        pickle.dump(max_len_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    agent2_model.save(model_path)
    #---------stop save---------------------
    agent2_model = tf.keras.models.load_model(model_path)
    x_test = x_train[0:10]
    preds = agent2_model.predict(np.array(x_test))
    #sort_1 = preds.argsort().tolist()
    #sort_2 = np.argmax(preds[0],axis=-1)
    preds_actual = []
    for x_sample in x_test:
        preds = agent2_model.predict(np.array([x_sample]))
        max_pred = np.argmax(preds,axis=-1)[0]
        preds_actual.append(reversed_action_vocab[max_pred])
    x_test2 = x_train[48:55]
    preds_actual2= []
    for x_sample in x_test2:
        preds = agent2_model.predict(np.array([x_sample]))
        max_pred = np.argmax(preds,axis=-1)[0]
        preds_actual2.append(reversed_action_vocab[max_pred])
    print(labels)
if __name__ == '__main__':
    print('start')
    #run_regular_pipe()
    agent1_start_time = time.time()
    agent1_save_path = 'dec_rock_essential/11_04_2023/agent1/'
    agent1_traces_path = 'dec_rock3x4x3_11_04_2023'
    agent1_bounds = [0,2,0,3]
    run_pipe_with_action_labels(agent1_save_path,agent1_traces_path,agent1_bounds)
    agent1_end_time = time.time()

    agent2_start_time = time.time()
    agent2_save_path = 'dec_rock_essential/11_04_2023/agent2/'
    agent2_traces_path = 'dec_rock3x4x3_11_04_2023'
    agent2_bounds = [1,4,0,3]
    run_pipe_with_action_labels_for_agent2(agent2_save_path,agent2_traces_path,agent2_bounds)
    agent2_end_time = time.time()

    print(f'agent 1 time : {agent1_end_time-agent1_start_time}  agent 2 time: {agent2_end_time-agent2_start_time}')
    #agent1_model = tf.keras.models.load_model('models/agent1_model_11_2.h5')
    #agent1_model.summary()
    #agent2_model = tf.keras.models.load_model('models/agent2_model_11_2.h5')
    #agent2_model.summary()


