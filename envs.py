import numpy as np
from scipy import io
import gym
from gym import spaces
from gym.utils import seeding
# from PPOPolicy import PPO
from logger import Logger

logger = Logger.get_logger(__name__)



class ShellEnv(gym.Env):
    def __init__(self, Tsim, Q, R, gamma, goal=1):
        """
        env initialization
        :param Tsim: length of simulation
        :param Q: error weighted matrix in MPC
        :param R: action weighted matrix in MPC
        :param gamma: reward discounted factor
        :param noise: whether to use noise
        """
        self.Tsim = Tsim
        self.Q = Q
        self.R = R
        self.gamma = gamma
        self.nu = 3
        self.ny = 3
        self.nd = 3  # add directly to output
        self.goal = goal * np.ones((self.ny, self.Tsim))
        self.num_step = 0
        self.action_bound = 0.05 * np.ones((self.nu, 1))
        self.total_reward = []

        self.qn, self.rn = int(self.Q.shape[0] / self.ny), int(self.R.shape[0] / self.nu)
        self.back = max(self.qn, self.rn)
    
        self.q, self.r = self._calc_qr(qn=self.qn, rn=self.rn, scaling=True)

        '''
        action = np.array([u])  # self.nu x 1 
        -uLimit <= action <= uLimit 
        '''
        self.action_space = spaces.Box(
            -np.ones((self.nu,)).astype(np.float32),
            np.ones((self.nu,)).astype(np.float32),
            shape=(self.nu,)
        )

        '''
        ================= observation ===============
        goal   |  [-10, 10]  |  (self.ny, self.back)
        error  |  [-1,   1]  |  (self.ny, self.back)
        action |  [-1 ,  1]  |  (self.nu, self.back)
        =============================================
        '''
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.ny*2+self.nu, self.back), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.ny+self.nu, self.back), dtype=np.float32)
        '''
        ** spaces.Dict is not supported in stable_baselines3 yet **
        '''
        # self.observation_space = spaces.Dict(
        #     {
        #         "goal": spaces.Box(low=-10, high=10, shape=(self.ny, self.back), dtype=np.float32),
        #         "error": spaces.Box(low=-10, high=10, shape=(self.ny, self.back), dtype=np.float32),
        #         "action": spaces.Box(low=-1, high=1, shape=(self.nu, self.back), dtype=np.float32)
        #     }
        # )

        '''
        state initial define
        '''
        goal = self.goal[:, -self.back:]
        error = self.goal[:, -self.back:]
        # error = (error - error.mean()) / (error.std() + 1e-8) + error.mean()
        action = np.zeros((self.nu, self.back))
        # self.state = np.vstack((goal, error, action))
        self.state = np.vstack((goal, error, action))
        # self.state = np.vstack((error, action))
        # self.state = {
        #     "goal": goal,
        #     "error": error,
        #     "action": action
        # }

        '''
        define params used to iteratively calculate the u, y, noise
        let S be the finite step response value of the transfer function 
        let M be the correction matrix in DMC
        then, Y(k) = M * Y(k-1) + S * du(k-1)
        '''
        self.S = io.loadmat('./step.mat')['Step']  # N x 3 x 3 (noise not included)
        self.N = self.S.shape[0]
        self.M = np.block([
            [np.zeros((self.N - 1, 1)), np.eye(self.N - 1)],
            [np.zeros((1, self.N - 1)), 1]
        ])
        self.yk = np.zeros((self.N, self.ny))
        self.y = np.zeros((self.ny, 1))
        self.du = np.zeros((self.nu, 1))
        self.u = np.zeros_like(self.du)

        self.seed()
        self.reset()

    def _calc_qr(self, qn, rn, scaling=True):
        '''
        Q'_i = \sum_{i=0}^{P-1} \gamma^i * q_i
        R'_i = \sum_{i=0}^{M-1} \gamma^i * r_i
        :return: q, r
        '''
        q, r = [], []
        for i in range(self.nu):  # R accordingly
            g = np.logspace(0, rn - 1, num=rn, base=self.gamma)
            r.append(np.sum(g * np.diag(self.R[i * rn:(i + 1) * rn, i * rn:(i + 1) * rn])))
        for i in range(self.ny):  # Q accordingly
            g = np.logspace(0, qn - 1, num=qn, base=self.gamma)
            q.append(np.sum(g * np.diag(self.Q[i * qn:(i + 1) * qn, i * qn:(i + 1) * qn])))
        q, r = np.array(q).reshape(1, -1), np.array(r).reshape(1, -1)
        if scaling:
            scale_factor = max(np.max(q), np.max(r))
            q /= scale_factor
            r /= scale_factor
        return q, r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.num_step += 1
        '''
        env state update 
        '''
        self.du = action.reshape(-1, 1) * self.action_bound  # shape (self.nu, 1)
        # self.du = action.reshape(-1, 1)
        self.u += self.du
        self.yk = self.M.dot(self.yk)
        for i in range(self.ny):
            self.yk[:, i] += self.S[:, i, :].dot(self.du).squeeze()
        self.y = self.yk[0, :]

        e = (self.goal[:, self.num_step - 1] - (self.y + self.noise[self.num_step - 1, :])).reshape(self.ny,)
        # e = (e - e.mean()) / (e.std() + 1e-8) + e.mean()
        '''
        update state 
        '''
        current_state = np.vstack(
            (
                self.goal[:, self.num_step - 1].reshape(self.ny, 1),
                e.reshape(self.ny, 1),
                action.reshape(self.nu, 1)
            )
        )
        self.state = np.hstack(
            (
                self.state[:, 1:],  # element from index = 1 to last
                current_state
            )
        )

        # self.state["goal"] = np.hstack((self.state["goal"][:, :-1], self.goal[:, self.num_step - 1].reshape(self.ny, )))
        # self.state["error"] = np.hstack((self.state["error"][:, :-1], e))
        # self.state["action"] = np.hstack((self.state["action"][:, :-1], action))

        reward = 0
        done = False
        '''
        done condition
        '''
        if self.num_step == self.Tsim:
            done = True

        '''
        reward function 
        '''
        # self.q = np.array([1, 1, 1])
        # self.r = np.array([1, 1, 1])
        reward -= np.sum(self.q.dot(e**2) + self.r.dot(self.du**2))
        # reward -= np.sum(self.q.dot(np.abs(e)) + self.r.dot(np.abs(self.du)))

        self.total_reward[-1] += reward

        return self.state, reward, done, {}

    def reset(self):
        goal = self.goal[:, -self.back:]
        error = self.goal[:, -self.back:]
        # error = (error - error.mean()) / (error.std() + 1e-8) + error.mean()
        action = np.zeros((self.nu, self.back))
        self.state = np.vstack((goal, error, action))
        # self.state = np.vstack((error, action))
        # self.state = {
        #     "goal": goal,
        #     "error": error,
        #     "action": action
        # }

        self.num_step = 0
        self.total_reward.append(0)

        self.yk = np.zeros((self.N, self.ny))
        self.y = np.zeros((self.ny, 1))
        self.du = np.zeros((self.nu, 1))
        self.u = np.zeros_like(self.du)

        self.noise = np.zeros((self.Tsim, self.ny))
        # self.noise = np.random.normal(0, 0.1, (self.Tsim, self.ny))
        return self.state


if __name__ == "__main__":
    # P = 30
    # M = 5
    # # base_Q = scipy.linalg.block_diag(np.eye(P), np.eye(P), np.eye(P))
    # # base_R = scipy.linalg.block_diag(np.eye(M), np.eye(M), np.eye(M))
    # base_Q = np.eye(P)
    # base_R = np.eye(M)
    # env = CSTR(
    #     Tsim = 100,
    #     Q = base_Q,
    #     R = base_R,
    #     gamma = 0.97,
    #     args= CSTR_Params()
    # )

    # env = ShellEnv(
    #     Tsim=100,
    #     Q=base_Q,
    #     R=base_R,
    #     gamma=0.97
    # )
    pass 
