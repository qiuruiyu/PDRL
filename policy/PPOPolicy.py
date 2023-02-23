import os
import sys
sys.path.append('./')
from collections import namedtuple

import gym
import numpy as np
import pandas as pd
import scipy.linalg
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from config import Params
from logger import Logger
from tensorboardX import SummaryWriter
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import criterion, save_pickle, make_env

from policy.worker import MemorySampler


sns.set_style("whitegrid")

logger = Logger.get_logger(__name__)

Transition = namedtuple('Transition', ['s', 'a', 'r', 'log_prob', 'd', 's_'])


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, action_low, action_high):
        super(Actor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.action_low = action_low
        self.action_high = action_high
        self.fc1 = nn.Linear(self.in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(128, 64)
        self.mu_head = nn.Linear(64, self.out_dim)
        self.sigma_head = nn.Linear(64, self.out_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        nn.init.orthogonal_(self.fc3.weight, gain=1)
        nn.init.orthogonal_(self.mu_head.weight, gain=1)
        nn.init.orthogonal_(self.sigma_head.weight, gain=1)

    def forward(self, x):
        if x.ndim >= 2:
            x = x.flatten(x.ndim-2, -1).reshape(-1, self.in_dim)
        scale = (self.action_high - self.action_low) / 2
        bias = scale - self.action_high
        x = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        # x = torch.tanh(self.fc1((x)))
        mu = scale * torch.tanh(self.mu_head(x)) - bias
        # mu = scale * 2 * (torch.sigmoid(self.mu_head(x)) - 0.5) - bias
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma


class BoundedActor(Actor):
    def __init__(self, in_dim, out_dim, action_low, action_high):
        super(BoundedActor, self).__init__(in_dim, out_dim, action_low, action_high)

    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        scale = (self.action_high - self.action_low) / 2
        bias = scale - self.action_high
        scale, bias = torch.FloatTensor(scale), torch.FloatTensor(bias)
        x = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        mu = self.mu_head(x)
        # mu = torch.max(torch.min(mu, torch.FloatTensor(self.action_high)), torch.FloatTensor(self.action_low))
        # mu = scale * mu - bias
        # mu = scale * 2 * (torch.sigmoid(self.mu_head(x)) - 0.5) - bias
        return mu


class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        nn.init.orthogonal_(self.state_value.weight, gain=1)

    def forward(self, x):
        if x.ndim >= 2:
            x = x.flatten(x.ndim-2, -1).reshape(-1, self.in_dim)
        x = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        # x = torch.tanh(self.fc1(x))
        value = self.state_value(x)
        return value


class RNNActor(nn.Module):
    def __init__(self, in_dim, out_dim, action_low, action_high):
        super(RNNActor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.action_low = action_low
        self.action_high = action_high

        self.rnn_layer = nn.RNN(
            input_size=9,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.rnn_layer.hidden_size, 8)
        self.mu_head = nn.Linear(8, self.out_dim)
        self.sigma_head = nn.Linear(8, self.out_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.mu_head.weight, gain=1)
        nn.init.orthogonal_(self.sigma_head.weight, gain=1)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = torch.permute(x, (0, 2, 1))
        scale = (self.action_high - self.action_low) / 2
        bias = scale - self.action_high
        r_out, hn = self.rnn_layer(x)
        x = torch.tanh(self.fc1(r_out[:, -1, :]))
        mu = scale * torch.tanh(self.mu_head(x)) - bias
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma


class RNNCritic(nn.Module):
    def __init__(self, in_dim):
        super(RNNCritic, self).__init__()
        self.in_dim = in_dim
        self.rnn_layer = nn.RNN(
            input_size=9,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.rnn_layer.hidden_size, 64)
        self.state_value = nn.Linear(64, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.state_value.weight, gain=1)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = torch.permute(x, (0, 2, 1))
        r_out, hn = self.rnn_layer(x)
        x = torch.tanh(self.fc1(r_out[:, -1, :]))
        x = self.state_value(x)
        return x


class Policy(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            action_bound_low,
            action_bound_high,
            args: Params):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_low = torch.tensor(action_bound_low)
        self.action_bound_high = torch.tensor(action_bound_high)
        self.args = args

        self.Anet = Actor(self.state_dim, self.action_dim, self.action_bound_low, self.action_bound_high).to(self.args.device)
        self.Cnet = Critic(self.state_dim).to(self.args.device)
        
        # self.Anet = RNNActor(self.state_dim, self.action_dim, self.action_bound_low, self.action_bound_high).to(
        #     self.args.device
        # )
        # self.Cnet = RNNCritic(self.state_dim).to(self.args.device)

        self.optim_a = optim.Adam(self.Anet.parameters(), lr=self.args.lr_a, eps=self.args.EPS)
        self.optim_c = optim.Adam(self.Cnet.parameters(), lr=self.args.lr_c, eps=self.args.EPS)

    def choose_action(self, s):
        if isinstance(s, int):
            s = np.array([s], dtype=np.float32)
        s = torch.FloatTensor(s).to(self.args.device)
        with torch.no_grad():
            mean, std = self.Anet(s)
        pd = Normal(mean, std)
        action = pd.sample()
        # action = torch.clamp(action[0], self.action_bound_low, self.action_bound_high)
        action = torch.max(torch.min(action[0], self.action_bound_high), self.action_bound_low)  # subsititute for clamp in torch_v_1.8.1
        action_log_prob = pd.log_prob(action)
        return action.cpu().numpy(), action_log_prob.cpu().numpy()

    def get_value(self, s):
        if isinstance(s, int):
            s = np.array([s], dtype=np.float32)
        s = torch.FloatTensor(s).to(self.args.device)
        with torch.no_grad():
            value = self.Cnet(s)
        return value

    def update_net(self, total_loss, i_episode):
        # linear schedule
        for p in self.optim_a.param_groups:
            p["lr"] = self.args.lr_a * (1 - i_episode / self.args.num_episode)
        for p in self.optim_c.param_groups:
            p["lr"] = self.args.lr_c * (1 - i_episode / self.args.num_episode)
        self.optim_a.zero_grad()
        self.optim_c.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.Anet.parameters(), self.args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.Cnet.parameters(), self.args.max_grad_norm)
        self.optim_a.step()
        self.optim_c.step()


class PPO(object):
    def __init__(
            self,
            env: gym.Env,
            q: int, 
            r: int, 
            args: Params,
            logger: Logger):
        self.env = env
        self.args = args
        self.logger = logger
        self.q, self.r = q, r 

        # make directory for data storage
        if self.args.save_data: 
            self._mkdir()

        if len(self.env.observation_space.shape) > 1:
            self.state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        else:
            self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound_high = env.action_space.high
        self.action_bound_low = env.action_space.low

        # define Policy here
        self.policy = Policy(self.state_dim, self.action_dim, self.action_bound_low, self.action_bound_high, self.args)
        self.BoundedAnet = BoundedActor(self.state_dim, self.action_dim, self.action_bound_low, self.action_bound_high).to(self.args.device)
        dummy_input = torch.rand(1, self.state_dim)
        self.model = BoundedModule(self.BoundedAnet, (dummy_input, ))

        if self.args.net_structure_log:
            if not os.path.exists(self.args.net_structure_logger_path):
                os.makedirs(self.args.net_structure_logger_path + 'ActorNet')
                os.makedirs(self.args.net_structure_logger_path + 'CriticNet')
            with SummaryWriter(self.args.net_structure_logger_path + 'ActorNet') as w:
                # w.add_graph(self.policy.Anet, torch.zeros(self.env.observation_space.shape))
                w.add_graph(self.policy.Anet, torch.zeros(1, 9, 30))
            with SummaryWriter(self.args.net_structure_logger_path + 'CriticNet') as w:
                # w.add_graph(self.policy.Cnet, torch.zeros(self.env.observation_space.shape))
                w.add_graph(self.policy.Cnet, torch.zeros(1, 9, 30))

        self.loss_fn = nn.MSELoss()

        # useful params
        self.total_step = 0
        self.sampler = None
        self.global_sample_size = 0
        self.policy_loss = []
        self.value_loss = []
        self.entropy = []
        self.kl_div = []
        self.reward = []
        self.running_reward = []
        self.best_reward = -np.inf
        self.best_episode_index = 0 

    def _mkdir(self):
        self.name = 'q_' + str(self.q) + '_r_' + str(self.r)
        # if self.args.adversary_attack:
        #     self.name = 'adv_' + self.name
        if self.args.adversary_attack:
            self.path = self.args.data_path + 'adv_' + self.name
        else:
            self.path = self.args.data_path + self.name
        self.model_path = os.path.join(self.path, 'model')
        # self.logger.info('config save path to {}'.format(os.path.join(self.path, 'model')))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _get_summary_writer(self):
        # prefix = "PPO_"
        # suffix = 0
        if not os.path.exists(self.args.tensorboard_logger_path):
            os.makedirs(self.args.tensorboard_logger_path)
        # n = len(os.listdir(self.args.tensorboard_logger_path))
        # if n > 0:
        #     suffix = n
        # logger_name = prefix + str(suffix)
        if self.args.adversary_attack:
            self.writer = SummaryWriter(logdir=os.path.join(self.args.tensorboard_logger_path, 'adv_' + self.name))
        else:
            self.writer = SummaryWriter(logdir=os.path.join(self.args.tensorboard_logger_path, self.name))
        return self.writer

    def _calc_advs(self, s, s_, r, d):
        """
        Calculate future discounted value according to Bellman Equation
        Return a batch of list v_targets and gae_advantages
        """
        # s_ = s_.reshape(-1, self.state_dim)
        # s = torch.flatten(s, start_dim=1, end_dim=-1)
        with torch.no_grad():
            v = self.policy.Cnet(s).reshape(-1, 1)
            v_ = self.policy.Cnet(s_[-1]).reshape(-1, 1)
        v_all = torch.cat((v, v_), dim=0)
        advs = torch.zeros_like(v)
        future_advs = 0.0
        for t in reversed(range(len(v))):
            delta = r[t] + self.args.gamma * v_all[t + 1] * (1 - d[t]) - v_all[t]
            advs[t] = future_advs = delta + self.args.gamma * self.args.gae_lam * (1 - d[t]) * future_advs
        v_targets = advs + v
        # norm advs
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return v, v_targets, advs

    def _save_model(self):
        torch.save(self.policy.Anet.state_dict(), self.path+'/model/Anet.pth')
        torch.save(self.policy.Cnet.state_dict(), self.path+'/model/Cnet.pth')

    def _load_model(self):
        name = 'q_' + str(self.q) + '_r_' + str(self.r)
        if self.args.adversary_attack:
            name = 'adv_' + name
        tmp_path = self.args.data_path + name
        tmp_model_path = os.path.join(tmp_path, 'model')
        self.logger.info("model load from: {}".format(tmp_model_path))
        Anet_state_dict = torch.load(tmp_model_path + '/' + 'Anet.pth')  
        Cnet_state_dict = torch.load(tmp_model_path + '/' + 'Cnet.pth')
        self.policy.Anet.load_state_dict(Anet_state_dict, strict=True)
        self.policy.Cnet.load_state_dict(Cnet_state_dict, strict=True)

    @staticmethod
    def _intermediate_to_kl(lb, ub, means, stdev=None):
        lb = lb - means
        ub = ub - means 
        u = torch.max(lb.abs(), ub.abs())
        if stdev is None: 
            return ((u * u).sum(axis=-1, keepdim=True))
        else:
            return ((u * u) / (stdev * stdev)).sum(axis=-1, keepdim=True)

    def _calc_upper_bound(self, model, x, means):
        """
        Calculate the error upper bound of the model 
        working as the regularizer of loss
        Create a BoundedActor instance to just calculate the \mu_\hat{s}
        """
        x = BoundedTensor(x, ptb=PerturbationLpNorm(norm=np.inf, eps=0.1)).requires_grad_(False)
        # forward one, 'backward' --> the tightest bound
        inputs = (x, )
        lb, ub = model.compute_bounds(x=inputs, method='backward', bound_lower=True, bound_upper=True)
        kl = self._intermediate_to_kl(lb, ub, means)
        return kl

    def worker_terminate(self):
        if self.sampler is not None:
            workers = self.sampler.workers
            for worker in workers: 
                if worker.is_alive:
                    worker.terminate()
                    worker.join()

    def train(self):
        self.sampler = MemorySampler(self.args, self.logger)
        if self.args.tensorboard_log:
            self.writer = self._get_summary_writer()
            
        for i_episode in range(self.args.num_episode):
            memory = self.sampler.sample(self.policy)
            batch = memory.sample()
            batch_size = len(memory)
            self.global_sample_size += batch_size

            s = torch.tensor(np.array(batch.s), dtype=torch.float).to(self.args.device)
            a = torch.tensor(np.array(batch.a), dtype=torch.float).view(-1, self.action_dim).to(self.args.device)
            r = torch.tensor(np.array(batch.r), dtype=torch.float).view(-1, 1).to(self.args.device)
            log_prob = torch.tensor(np.array(batch.log_prob), dtype=torch.float).view(-1, self.action_dim).to(self.args.device)
            d = torch.tensor(np.array(batch.d), dtype=torch.int).view(-1, 1).to(self.args.device)
            s_ = torch.tensor(np.array(batch.s_), dtype=torch.float).to(self.args.device)

            # advs --> gae
            r = (r - torch.mean(r)) / (torch.std(r) + 1e-8)
            # r /= (torch.std(r) + 1e-8)

            v, v_targets, advs = self._calc_advs(s, s_, r, d)
            '''
            mini-batch learning 
            '''
            episode_policy_loss = []
            episode_value_loss = []
            episode_total_loss = []
            episode_entropy = []
            episode_kl_div = []
            episode_upper_kl_loss = [] 
            for k in range(self.args.num_epoch):
                for index in BatchSampler(
                        SubsetRandomSampler(range(self.args.batch_size)), self.args.mini_batch_size, False
                ):
                    mean, std = self.policy.Anet(s[index])
                    # print('mean', mean)
                    # print('std', std)
                    # print('--------------------')
                    new_pd = Normal(mean, std)
                    action_log_prob = new_pd.log_prob(a[index])
                    entropy_loss = -action_log_prob.mean()
                    self.entropy.append(-entropy_loss.item())
                    episode_entropy.append(-entropy_loss.item())
                    ratio = torch.exp(action_log_prob - log_prob[index])
                    with torch.no_grad():
                        log_ratio = action_log_prob - log_prob[index]
                        kl = torch.mean((torch.exp(log_ratio)-1) - log_ratio).cpu().numpy()
                        if self.args.target_kl is not None:
                            if kl > self.args.target_kl:
                                self.logger.info("-----episode passed for too larget kl divergence-----")
                                break
                        self.kl_div.append(kl)
                        episode_kl_div.append(kl)
                    surr1 = ratio * advs[index]
                    surr2 = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip) * advs[index]
                    pol_loss = -torch.min(surr1, surr2).mean()
                    self.policy_loss.append(pol_loss.item())
                    episode_policy_loss.append(pol_loss.item())
                    val_loss = self.loss_fn(self.policy.Cnet(s[index]), v_targets[index]).mean()
                    self.value_loss.append(val_loss.item())
                    episode_value_loss.append(val_loss.item())
                    total_loss = pol_loss + self.args.loss_value_coef * val_loss + self.args.loss_entropy_coef * entropy_loss
                    if self.args.adversary_attack:
                        self.model.load_state_dict(self.policy.Anet.state_dict())
                        inputs = s[index].flatten(1, -1)
                        means = self.model(inputs).detach()
                        kl_loss = self._calc_upper_bound(self.model, inputs, means).mean().item() * self.args.loss_upper_kl_coef
                        episode_upper_kl_loss.append(kl_loss)
                        total_loss += kl_loss
                    episode_total_loss.append(total_loss.item())
                    self.policy.update_net(total_loss, i_episode)
                    if self.args.tensorboard_log:
                        self.writer.add_scalar("training parmas/lr_a", self.policy.optim_a.param_groups[-1]["lr"], global_step=i_episode)
                        self.writer.add_scalar("training parmas/lr_c", self.policy.optim_c.param_groups[-1]["lr"], global_step=i_episode)
            '''
            test part 
            '''
            test_rwd = []
            for _ in range(self.args.num_test):
                s = self.env.reset()
                epi_rwd = 0
                while True:
                    a, log_prob = self.policy.choose_action(s)
                    s_, r, done, _ = self.env.step(a)
                    epi_rwd += r
                    if done:
                        break
                    s = s_
                test_rwd.append(epi_rwd)
            self.reward.append(np.mean(test_rwd))
            if self.reward[-1] > self.best_reward:
                self.best_reward = self.reward[-1]
                self.best_episode_index = i_episode
                if self.args.save_data:
                    self._save_model()
            else:
                if self.args.early_stop and i_episode > self.best_episode_index + self.args.early_stop_episode:
                    self.logger.info("Early stop at episode: {}".format(i_episode))
                    self.logger.info("Best reward is: {:.4f}".format(self.best_reward))
                    break

            if len(self.running_reward) == 0:
                self.running_reward.append(np.mean(test_rwd))
            else:
                self.running_reward.append(self.running_reward[-1] * 0.9 + np.mean(test_rwd) * 0.1)

            self.logger.info(
                "----------------------" + str(i_episode) + "-------------------------"
            )
            self.logger.info("Episode reward: {:.4f}".format(self.reward[-1]))
            self.logger.info("Best reward: {:.4f}".format(self.best_reward))
            self.logger.info("Running reward: {:.4f}".format(self.running_reward[-1]))
            self.logger.info("Total loss: {:.4f}".format(np.mean(episode_total_loss)))
            self.logger.info("Policy loss: {:.4f}".format(np.mean(episode_policy_loss)))
            self.logger.info("Value loss: {:.4f}".format(np.mean(episode_value_loss)))
            self.logger.info("upper kl loss: {:.4f}".format(np.mean(episode_upper_kl_loss)))
            self.logger.info("Entropy: {:.4f}".format(np.mean(episode_entropy)))
            self.logger.info("KL_divergence: {:.4f}".format(np.mean(episode_kl_div)))
            self.logger.info("Total sample size: {}".format(self.global_sample_size))
            self.logger.info("Actor learning rate: {}".format(self.policy.optim_a.param_groups[-1]["lr"]))
            self.logger.info("Critic learning rate: {}".format(self.policy.optim_c.param_groups[-1]["lr"]))

            if self.args.tensorboard_log:
                self.writer.add_scalar("reward/episode reward", self.reward[-1], global_step=i_episode)
                self.writer.add_scalar("reward/best reward", self.best_reward, global_step=i_episode)
                self.writer.add_scalar("loss/total loss", np.mean(episode_total_loss), global_step=i_episode)
                self.writer.add_scalar("loss/policy loss", np.mean(episode_policy_loss), global_step=i_episode)
                self.writer.add_scalar("loss/value loss", np.mean(episode_value_loss), global_step=i_episode)
                self.writer.add_scalar("loss/entropy", np.mean(episode_entropy), global_step=i_episode)
                self.writer.add_scalar("loss/kl divergence", np.mean(episode_kl_div), global_step=i_episode)
                self.writer.add_scalar("loss/upper bound kl dist", np.mean(episode_upper_kl_loss), global_step=i_episode)

            if self.args.save_data:
                training_pickle = {
                    'reward': self.reward,
                    'policy_loss': self.policy_loss,
                    'value_loss': self.value_loss,
                    'entropy': self.entropy,
                    'kl_div': self.kl_div,
                }
                save_pickle(training_pickle, os.path.join(self.path, 'train_record.pkl'))

    def test(self, noise=None):
        # check path 
        if self.args.save_data: 
            if not os.path.exists(self.args.data_path):
                os.mkdir(self.args.data_path)
        if self.args.adversary_attack:
            name = 'adv_' + self.name
        else:
            name = self.name 
        path = self.args.data_path + name

        data = []
        reward = []
        IAE, ISE, ITAE, ITSE = [], [], [], []
        
        self._load_model()

        for i in range(self.args.num_test):
            # self.env = make_env("1", self.q, self.r)
            # self.env.seed(self.args.seed + i)
            s = self.env.reset()
            if self.args.adversary_attack_var:
                attacked_s = s.copy()
                # attacked_s[3:6, -1] = attacked_s[:3, -1] - np.random.normal((attacked_s[:3, -1] - attacked_s[3:6, -1]), self.args.adversary_attack_var)
                attacked_s[3:6, -1] = attacked_s[3:6, -1] + noise[i, 0, :].reshape(attacked_s[3:6, -1].shape)
            epi_rwd = 0 
            step = 0
            while True:
                step += 1
                actual_output = (1 - s[3:6, -1])
                '''
                ['type', 'value', 'num_test', 'index', 'step']
                '''
                data.append(['y', actual_output[0], i + 1, 1, step])
                data.append(['y', actual_output[1], i + 1, 2, step])
                data.append(['y', actual_output[2], i + 1, 3, step])
                if self.args.adversary_attack_var:
                    a, log_prob = self.policy.choose_action(attacked_s)
                else:
                    a, log_prob = self.policy.choose_action(s)
                data.append(['a', a[0], i + 1, 1, step])
                data.append(['a', a[1], i + 1, 2, step])
                data.append(['a', a[2], i + 1, 3, step])

                s_, r, done, _ = self.env.step(a)
                
                if self.args.adversary_attack_var:
                    attacked_s_ = s_.copy()
                    # attacked_s_[3:6, -1] = attacked_s_[:3, -1] - np.random.normal((attacked_s_[:3, -1] - attacked_s_[3:6, -1]), self.args.adversary_attack_var)
                    attacked_s_[3:6, -1] = attacked_s_[3:6, -1] + noise[i, step-1, :].reshape(attacked_s_[3:6, -1].shape)
                    attacked_s = np.hstack((
                        attacked_s[:, 1:],
                        attacked_s_[:, -1].reshape(-1, 1)
                    ))
                    # attacked_s = attacked_s_
                
                epi_rwd += r
                if done:
                    break
                s = s_
            reward.append(epi_rwd)
        print(reward)
        
        df = pd.DataFrame(data=data, columns=['type', 'value', 'num_test', 'index', 'step'])
        df.sort_values(by=['num_test', 'step'])  # 1st: num_test, 2nd: index, 3rd: step

        for i in range(self.args.num_test):
            df_y = df[df['type'] == 'y']
            test_y = df_y[df_y['num_test'] == i + 1]['value'].to_numpy().reshape(self.env.Tsim, self.env.ny)
            # test_y = df[df['num_test'] == i + 1]['y'].to_numpy().reshape(self.env.Tsim, self.env.ny)
            iae, ise, itae, itse = criterion(test_y, 1)
            IAE.append(iae)
            ISE.append(ise)
            ITAE.append(itae)
            ITSE.append(itse)
        
        if self.args.save_data:
            test_pickle = {
                'data_df': df,
                'reward': reward,
                'IAE': IAE,
                'ISE': ISE,
                'ITAE': ITAE,
                'ITSE': ITSE,
            }
            
            save_pickle(test_pickle, os.path.join(path, 'test_record.pkl'))
            logger.info('Successfully save pickle to {}'.format(os.path.join(path, 'test_record.pkl')))


if __name__ == "__main__":
    var = np.concatenate((
        0 * np.ones((10, 15, 3)),
        0 * np.ones((10, 85, 3)),
    ),axis=1)
    noise = np.random.normal(0, var)

    # try:
    #     # q, r = Params.q, Params.r
    #     q, r = 1, 10
    #     env = make_env("1", q=q, r=r)
    #     ppo = PPO(env=env, logger=logger, q=q, r=r, args=Params)
    #     ppo.train()
    #     # ppo.test(q=q, r=r, adv=False, noise=noise)
        
    # except Exception as e:
    #     print(f"exception: {traceback.print_exc()}")

    q_r_pair = [
        # (20, 1), (10, 1), (1, 1), (1, 3), (1, 5), (1, 7)
        (1, 1), (1, 3), (1, 5), (1, 10)
    ]

    for adv in [False]:
        for q, r in q_r_pair:
            env = make_env("1", q=q, r=r)
            args = Params(adversary_attack=adv)
            ppo = PPO(env=env, logger=logger, q=q, r=r, args=args)
            ppo.test(noise=noise)
