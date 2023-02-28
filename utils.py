import pickle
from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt 
from envs import ShellEnv
import scipy.linalg


def make_env(id: str, q=1, r=1, goal=1, seed=None):
    """
    Return env, and params: q, r 
    """
    P = 30
    M = 10
    base_Q = scipy.linalg.block_diag(np.eye(P), np.eye(P), np.eye(P))
    base_R = scipy.linalg.block_diag(np.eye(M), np.eye(M), np.eye(M))
    env = ShellEnv(
        Tsim=100,
        Q=base_Q * q,
        R=base_R * r,
        gamma=0.95,
        goal=goal,
    )

    return env


def obs_scaling(x):
    x = (x - np.mean(x, axis=-1)) / (np.std(x, axis=-1))
    return x


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_pickle(path):
    with open(path, 'rb') as f:
        try:
            res = pickle.load(f)
            return res
        except Exception as e:
            print('pickle load error')


def criterion(data, target):
    Tsim, n_var = data.shape
    t = np.arange(1, Tsim + 1).reshape(1, -1)
    e = np.abs(target - data)
    IAE = np.sum(e)
    ISE = np.sum(e * e)
    ITAE = np.sum(t.dot(e))
    ITSE = np.sum(t.dot(e * e))
    return IAE, ISE, ITAE, ITSE


def reward_cutting(step:Union[List, np.ndarray], reward: Union[List, np.ndarray], cutting_point: int = -800):
    '''
    A method used to re-cut the reward array so that the reward curve looks better 
    '''
    if not isinstance(reward, np.ndarray):
        res = np.array(reward)
    else: 
        res = reward 
    idx = np.where(res >= cutting_point)
    step = step[idx]
    step -= step[0]
    res = res[idx]
    return step, res


def filter_data(step, value, avg_win=20):
    res_step, rwd, std = [], [], [] 
    for i in range(len(value) - avg_win):
        res_step.append(step[i])
        rwd.append(np.mean(value[i:i+avg_win]))
        std.append(np.std(value[i:i+avg_win]))
    return res_step, rwd, std
    # for i in range(len(data) - avg_win):
    #     iter.append(i+1)
    #     rwd.append(np.sum(data[i:i+avg_win])/avg_win)
    #     std.append(np.std(data[i:i+avg_win]))
    # return iter, rwd, std     


def reward_plot(iter, rwd, std, color, dpi=100, type='normal'):
    idx, max_rwd = rwd.index(max(rwd)), max(rwd)
    if type == 'normal':
        plt.figure(figsize=(24, 12), dpi=dpi)
        plt.plot(iter, rwd, color=color)
        # plt.scatter(idx, max_rwd, color='r')
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Reward', fontsize=20)
        plt.title('Filtered Reward', fontsize=20)
        plt.show()
    elif type == 'std':
        plt.figure(figsize=(24, 12), dpi=dpi)
        plt.plot(rwd, color=color)
        # plt.scatter(idx, max_rwd, color='r')
        r1 = list(map(lambda x: x[0]-x[1], zip(rwd, std)))
        r2 = list(map(lambda x: x[0]+x[1], zip(rwd, std)))
        plt.fill_between(iter, r1, r2, color=color, alpha=0.2)
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Reward', fontsize=20)
        plt.title('reward curve of training', fontsize=20)
        plt.show()       


def calc_dist(qr1: Tuple, qr2: Tuple, method='log'):
    '''
    A method used to measure the distance between 2 Q-R pairs 
    In representation of Q-R pairs, Q is set as 1, and R is changed to set the aggressiveness of controllers
    Param: method, must be in ['log', 'eud', ...] 
    '''
    q1, r1 = qr1
    q2, r2 = qr2
    # normalize q, r to q = 1 
    q1, r1 = 1, r1 / q1 
    q2, r2 = 1, r2 / q2

    '''
    log method can effectively magnify the difference when r is far smaller than 1  
    '''
    if method == 'log':
        dist = np.abs(np.log(r1) - np.log(r2))
    elif method == 'eud':
        dist = np.abs(r1 - r2)
    return dist 


