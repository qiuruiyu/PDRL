import multiprocessing as mp
import torch
import numpy as np
from collections import namedtuple
from logger import Logger
from config import Params
from utils import make_env


Transition = namedtuple('Transition', ['s', 'a', 'r', 'log_prob', 'd', 's_'])
Get_Enough_Batch = mp.Value("i", 0)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape: the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


class Episode(object):
    def __init__(self):
        self.episode = []

    def push(self, *args):
        """
        push transition of each step in an episode into the object
        :param args: ['s', 'a', 'r', 'log_prob', 'd', 's_']
        :return: None
        """
        self.episode.append(Transition(*args))

    def __len__(self):
        return len(self.episode)


class Memory(object):
    def __init__(self):
        self.memory = []
        self.num_episode = 0

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class EnvWorker(mp.Process):
    def __init__(self, remote, queue, lock, seed, worker_index, args: Params):
        super(EnvWorker, self).__init__()
        self.worker_index = worker_index
        self.remote = remote
        self.queue = queue
        self.lock = lock
        self.seed = seed
        self.args = args

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        """
        core running code for each env worker
        :return: None
        """
        self.env = make_env(self.worker_index, self.args.q, self.args.r, self.seed)
        # print('current running: env.q: {}, env.r: {}'.format(self.env.q, self.env.r))
        env_pid = -1
        while True:
            command, policy = self.remote.recv()
            if command == "sample":
                while Get_Enough_Batch.value == 0:
                    episode = Episode()
                    s = self.env.reset()
                    if self.args.adversary_attack_var:
                        attacked_s = s.copy()
                        # default the goal is 1 here 
                        attacked_s[1, -1] = attacked_s[0, -1] - np.random.normal((attacked_s[0, -1] - attacked_s[1, -1]), self.args.adversary_attack_var)
                    while Get_Enough_Batch.value == 0:
                        if not self.args.adversary_attack_var:
                            a, log_prob = policy.choose_action(s)
                            # v = policy.get_value(s)
                            # a = a[0]
                            log_prob = log_prob[0]
                            # v = v[0]
                            s_, r, done, _ = self.env.step(a)
                            episode.push(
                                s, a, r, log_prob, done, s_
                            )
                        else:
                            a, log_prob = policy.choose_action(attacked_s)
                            # v = policy.get_value(s)
                            # a = a[0]
                            log_prob = log_prob[0]
                            # v = v[0]
                            s_, r, done, _ = self.env.step(a)
                            attacked_s_ = s_.copy()
                            attacked_s_[1, -1] = attacked_s_[0, -1] - np.random.normal((attacked_s_[0, -1] - attacked_s_[1, -1]), self.args.adversary_attack_var)
                            episode.push(
                                attacked_s, a, r, log_prob, done, attacked_s_
                            )
                            attacked_s = attacked_s_
                        if done:
                            with self.lock:
                                self.queue.put(episode)
                            break
                        s = s_
            elif command == "close":
                self.remote.close()
                self.env.close()


class MemorySampler(object):
    def __init__(self, args, logger: Logger):
        self.args = args
        self.logger = Logger
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.device = args.device

        self.queue = mp.Queue()
        self.lock = mp.Lock()

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_workers)])
        self.workers = [
            EnvWorker(remote, self.queue, self.lock, self.args.seed + index, index, self.args)
            for index, remote in enumerate(self.work_remotes)
        ]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        for remote in self.work_remotes:
            remote.close()

    def sample(self, policy):
        policy.Anet.to("cpu")
        policy.Cnet.to("cpu")
        memory = Memory()
        Get_Enough_Batch.value = 0
        for remote in self.remotes:
            remote.send(("sample", policy))

        while len(memory) < self.batch_size:
            episode = self.queue.get(True)
            memory.push(episode)

        Get_Enough_Batch.value = 1

        while self.queue.qsize() > 0:
            self.queue.get()

        policy.Anet.to(self.device)
        policy.Cnet.to(self.device)
        return memory

    def close(self):
        Get_Enough_Batch.value = 1
        for remote in self.remotes:
            remote.send(("close", None))
        for worker in self.workers:
            worker.join()

