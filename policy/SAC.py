import gym
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter 
import numpy as np 
import scipy.linalg
import matplotlib.pyplot as plt
import sys
import pandas as pd
from typing import Union
from collections import namedtuple
import time
from envs import ShellEnv
import seaborn as sns 
sys.path.append('./')


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.s_dim = s_dim 
        self.a_dim = a_dim 
        self.fc1 = nn.Linear(s_dim + a_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        nn.init.orthogonal_(self.out.weight, gain=1)

    def forward(self, s, a):
        if not isinstance(s, torch.FloatTensor):
            s = torch.tensor(s, dtype=torch.float32)
        if not isinstance(a, torch.FloatTensor):
            a = torch.tensor(a, dtype=torch.float32)
        if s.ndim >= 2:
            s = s.flatten(s.ndim-2, -1)
        s = s.to(device)
        a = a.to(device)
        self.combined = torch.cat((s, a), dim=-1)
        x = self.out(torch.tanh(self.fc2(torch.tanh(self.fc1(self.combined)))))
        return x


class Actor(nn.Module):
    """
    Squashed Gaussian MLP Actor 
    """
    def __init__(self, s_dim, a_dim, a_bound: tuple):
        super().__init__()
        self.low, self.high = a_bound
        self.scale = (self.high - self.low) / 2 
        self.bias = self.scale - self.high
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, a_dim)
        self.sigma = nn.Linear(64, a_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        nn.init.orthogonal_(self.mu.weight, gain=1)
        nn.init.orthogonal_(self.sigma.weight, gain=1)

    def forward(self, s):
        if not isinstance(s, torch.FloatTensor):
            s = torch.tensor(s, dtype=torch.float32)
        if s.ndim >= 2:
            s = s.flatten(s.ndim-2, -1)
        s = s.to(device)
        x = torch.tanh(self.fc1(s))
        mu = self.scale * torch.tanh(self.mu(x)) - self.bias
        sigma = F.softplus(self.sigma(x))
        return mu, sigma

    def choose_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
        pd = Normal(mu, sigma)
        action = pd.rsample() 
        action = torch.clamp(action, self.low, self.high)
        return action.cpu().numpy()

    def get_action_and_log_prob(self, s):
        mu, sigma = self.forward(s)
        pd = Normal(mu, sigma)
        action = pd.rsample()
        log_prob = torch.mean(pd.log_prob(action), dim=-1)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        action = torch.clamp(action, self.low, self.high)
        return action, log_prob


class Critic(nn.Module):
    def __init__(
        self,
        s_dim, 
        a_dim, 
        polyak: Union[None, float] = None):
        super().__init__()
        self.s_dim = s_dim 
        self.a_dim = a_dim 
        self.polyak = polyak

        self.c1 = CNet(self.s_dim, self.a_dim).to(device)
        self.c2 = CNet(self.s_dim, self.a_dim).to(device)

    def forward(self, s, a):
        q1 = self.c1(s, a)
        q2 = self.c2(s, a)
        return q1, q2


class Memory(object):
    def __init__(
        self,
        size,
        gamma, 
        s_dim,
        a_dim,  # origin size of action 
        batch_size,
        ) -> None:
        self.gamma = gamma
        self.size = size
        self.batch_size = batch_size
        self.s_dim = s_dim 
        self.a_dim = a_dim 

        # if isinstance(self.s_dim, tuple):
        #     self.s_buf = np.zeros((self.size, *(self.s_dim)))
        #     self.s_buf_ = np.zeros((self.size, *(self.s_dim)))
        # else:
        self.s_buf = np.zeros((self.size, self.s_dim))
        self.s_buf_ = np.zeros((self.size, self.s_dim))
        self.a_buf = np.zeros((self.size, self.a_dim))
        self.r_buf = np.zeros((self.size, 1))
        self.mask_buf = np.zeros((self.size, 1))
        
        self.counter = 0 
    
    def store_transition(self, s, a, r, d, s_):
        mask = self.gamma * (1 - d)
        index = self.counter % self.size 
        self.s_buf[index, :] = s
        self.a_buf[index, :] = a 
        self.r_buf[index, :] = r
        self.mask_buf[index, :] = mask
        self.s_buf_[index, :] = s_
        
        self.counter += 1

    def sample(self):
        assert self.counter >= self.size, 'Memory not filled'
        idx = np.random.choice(self.size, size=self.batch_size, replace=True)
        s = torch.FloatTensor(self.s_buf[idx])
        a = torch.FloatTensor(self.a_buf[idx])
        r = torch.FloatTensor(self.r_buf[idx])
        mask = torch.FloatTensor(self.mask_buf[idx])
        s_ = torch.FloatTensor(self.s_buf_[idx])
        return s, a, r, mask, s_


class SAC(object):
    def __init__(
        self,
        s_dim: Union[int, tuple],
        a_dim: int,
        a_bound: tuple,
        gamma: float = 0.99,
        size: int = 5000,  # size of replay buffer 
        batch_size: int = 32,  # batch size for per train 
        lr_a: int = 1e-3, 
        lr_c: int = 1e-3,
        lr_alpha: int = 1e-3,
        target_entropy: float = -1,  # auto for -|A| of env
        update_freq: int = 50,  # frequency of update step numbers 
        gradient_step: int = 20,  # number of gradient steps per learn 
        polyak: Union[None, float] = None,
        ) -> None:

        if isinstance(s_dim, tuple):
            self.ori_s_dim = s_dim
            self.s_dim = np.prod(np.array(s_dim))
        else:  # int type 
            self.s_dim = s_dim
        self.a_dim = a_dim 
        self.a_bound = a_bound 
        self.low, self.high = a_bound
        self.gamma = gamma 
        self.size = size 
        self.batch_size = batch_size 
        self.lr_a = lr_a 
        self.lr_c = lr_c
        self.lr_alpha = lr_alpha
        self.target_entropy = target_entropy
        self.update_freq = update_freq
        self.gradient_step = gradient_step
        self.polyak = polyak

        self.actor = Actor(
            self.s_dim, 
            self.a_dim,
            self.a_bound
        ).to(device)

        self.critic = Critic(
            self.s_dim,
            self.a_dim
        ).to(device)

        self.critic_target = Critic(
            self.s_dim,
            self.a_dim
        ).to(device)
        self.critic_target.train(False)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_a = optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optim_c = optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.alpha_log = torch.log(
            torch.ones(
                1, device=device
                ) * 1
            ).requires_grad_(True)
            
        self.optim_alpha = optim.Adam([self.alpha_log], lr=self.lr_alpha)

        self.memory = Memory(self.size, self.gamma, self.s_dim, self.a_dim, self.batch_size)

        # useful params
        self.total_step = 0 
        self.entropy = []
        self.entropy_losses = [] 
        self.policy_losses = [] 
        self.critic_losses = [] 

    def update_critic_params(self):
        if self.polyak is None: 
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def learn(self):
        for _ in range(self.gradient_step):
            s, a, r, mask, s_ = self.memory.sample()

            if hasattr(self, "ori_s_dim"):
                s = s.reshape(-1, *(self.ori_s_dim))
                s_ = s_.reshape(-1, *(self.ori_s_dim))

            # update critic 
            alpha = self.alpha_log.exp().detach()

            with torch.no_grad():
                a_, log_prob_ = self.actor.get_action_and_log_prob(s_)
                log_prob_ = log_prob_.reshape(-1, 1)
                q1_target, q2_target = self.critic_target(s_, a_)
                assert q1_target.shape == q2_target.shape == log_prob_.shape, 'q_target shape not equal to log_prob shape'
                q_ = torch.min(q1_target, q2_target) - alpha * log_prob_
                assert q_.shape == r.shape == mask.shape, 'calculate q_target, shape not equal'
                q_target = r + mask * q_
            q1, q2 = self.critic(s, a)
            critic_loss = 0.5 * (F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target))
            self.critic_losses.append(critic_loss.item())

            self.optim_c.zero_grad()
            critic_loss.backward()
            self.optim_c.step()
    
            # a, log_prob for alpha and actor update 
            a, log_prob = self.actor.get_action_and_log_prob(s)
            self.entropy.append(-log_prob.cpu().detach().numpy().mean())
    
            # update alpha
            log_prob = log_prob.reshape(-1, 1)
            alpha_loss = -(self.alpha_log * (log_prob + self.target_entropy).detach()).mean()      
            self.entropy_losses.append(alpha_loss.item())

            self.optim_alpha.zero_grad()
            alpha_loss.backward()
            self.optim_alpha.step() 

            # update actor
            q1, q2 = self.critic(s, a)
            q = torch.min(q1, q2)
            policy_loss = -(q - alpha * log_prob).mean()
            self.policy_losses.append(policy_loss.item())

            self.optim_a.zero_grad()
            policy_loss.backward()
            self.optim_a.step()

            self.update_critic_params()


P = 30
M = 5
base_Q = scipy.linalg.block_diag(np.eye(P), np.eye(P), np.eye(P))
base_R = scipy.linalg.block_diag(np.eye(M), np.eye(M), np.eye(M))
env = ShellEnv(
    Tsim=100,
    Q=base_Q,
    R=base_R,
    gamma=0.97
)
env.seed(1234)

# env = gym.make('Pendulum-v1')
s_dim = env.observation_space.shape
a_dim = env.action_space.shape[0]
a_bound = (-1, 1)


agent = SAC(
    s_dim = s_dim,
    a_dim = a_dim,
    a_bound = a_bound, 
    gamma = 0.99, 
    size = 100000,
    batch_size = 512, 
    lr_a = 1e-4,
    lr_c = 1e-4,
    lr_alpha = 3e-4,
    update_freq = 100,
    gradient_step = 30,
    target_entropy = -a_dim,
    polyak = 0.995 
)


rwd = []

for epi in range(4000):
    s = env.reset()
    epi_rwd = 0 

    while True:
        agent.total_step += 1
        if agent.memory.counter < agent.memory.size:
            a = env.action_space.sample()
        else:
            a = agent.actor.choose_action(s)
        s_, r, d, _ = env.step(a)
        agent.memory.store_transition(s.flatten(), a, r, d, s_.flatten())

        if agent.memory.counter >= agent.memory.size:
            if agent.total_step % agent.update_freq == 0:
                agent.learn()

        epi_rwd += r
        s = s_ 

        if d:
            break
    if agent.memory.counter >= agent.memory.size:
        rwd.append(epi_rwd)
    print(f'Episode: {epi}, total_reward: {epi_rwd}')

rwd_avg = []

win = 30
for i in range(len(rwd)-win):
    rwd_avg.append(np.mean(rwd[i:i+win]))

# plot visualize 
sns.set_style("whitegrid")
fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6))
ax0 = sns.lineplot(x=list(range(len(rwd))), y=rwd, color='b', ax=ax[0][0])
ax0 = sns.lineplot(x=list(range(len(rwd_avg))), y=rwd_avg, color='r', ax=ax[0][0])
ax0.set_title('reward')
ax1 = sns.lineplot(x=list(range(len(agent.critic_losses))), y=agent.critic_losses, ax=ax[0][1])
ax1.set_title('critic loss')
ax2 = sns.lineplot(x=list(range(len(agent.policy_losses))), y=agent.policy_losses, ax=ax[1][0])
ax2.set_title('policy loss')
ax3 = sns.lineplot(x=list(range(len(agent.entropy))), y=agent.entropy, ax=ax[1][1])
ax3.set_title('entropy')

plt.show()

# plt.figure(figsize=(6, 4), dpi=100)
# plt.plot(rwd, 'b')
# plt.plot(rwd_avg, 'r')
# plt.title('reward')

# plt.figure(figsize=(6, 4), dpi=100)
# plt.plot(agent.critic_losses)
# plt.title('critic loss')

# plt.figure(figsize=(6, 4), dpi=100)
# plt.plot(agent.policy_losses)
# plt.title('policy loss')

# plt.figure(figsize=(6, 4), dpi=100)
# plt.plot(agent.entropy)
# plt.title('entropy')

# plt.show() 
