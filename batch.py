from typing import List, Callable
from config import Params
from logger import Logger
# from policy.PPOPolicy import PPO 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.utils import Schedule, set_random_seed
from utils import make_env
from envs import ShellEnv
import numpy as np 
import scipy.linalg
from stable_baselines3.common.env_util import make_vec_env

logger = Logger.get_logger(__name__)  

param_list: List[Params] = [] 

num_workers = 10

q_r_pair = [
    (1, 1)
    # (1, 1), (1, 3), (1, 5), (1, 7), (1, 9)
]

num_episode = {
    1: 1200,
    0: 2000
}


def square_schedule(init_value: float):
    """
    Linear learning rate schedule.
    :param init_value: initial learning rate
    :return: schedule that computes current learning rate
    depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 to 0
        :param progress_remaining:
        :return: current learning rate
        """
        return init_value / (2**(1-progress_remaining))
    return func


if __name__ == "__main__":
    for adv in [False]: 
        for qr in q_r_pair:
            q, r = qr
            param = Params(
                q=q,
                r=r,
                num_workers=num_workers,
                num_episode=3000,
                adversary_attack=adv,
                loss_upper_kl_coef=0.005, 
                early_stop_episode=99999
            )
            param_list.append(param)

    for param in param_list:
        # try:
        q, r = param.q, param.r
        print('START TRAINIG: q - {}, r - {}'.format(q, r))
        # env = make_env("1", q, r)

        P = 30
        M = 10
        base_Q = scipy.linalg.block_diag(np.eye(P), np.eye(P), np.eye(P))
        base_R = scipy.linalg.block_diag(np.eye(M), np.eye(M), np.eye(M))

        env = make_env(
            id = 'train',
            q = q,
            r = r,
            goal = 1,
        )
#         env = make_vec_env(
#             ShellEnv,
#             n_envs=10,
#             env_kwargs={'Tsim': 100, 'Q':base_Q*q, 'R':base_R*r, 'gamma':0.95}
# )

        evalCallback = EvalCallback(
                eval_env=env,
                eval_freq=5000,
                n_eval_episodes=10,
                # callback_after_eval=stop_train_callback,
                log_path='./data/agents/q_'+str(q)+'_r_'+str(r)+'/Log',
                best_model_save_path='./data/agents/q_'+str(q)+'_r_'+str(r)+'/Evalpoint',
                deterministic=True,  
                render=False,
            )
        # stop_train_callback = StopTrainingOnNoModelImprovement(
        #         max_no_improvement_evals=100, 
        #         min_evals=300,
        #         verbose=1
        #     )
        callback = CallbackList([evalCallback])

        agent = PPO(
            policy='MlpPolicy',
            learning_rate=square_schedule(7e-4),
            env=env,
            verbose=1,
            device='cpu',
            tensorboard_log='./data/agents/tensorboard/q_'+str(q)+'_r_'+str(r),
            n_steps=1024
        )

        agent.learn(total_timesteps=5000000, callback=callback)
        agent.save('./data/agents/q_'+str(q)+'_r_'+str(r)+'/model')

        # ppo = PPO(
        #     env = env,
        #     logger = logger, 
        #     q = q,
        #     r = r, 
        #     args = Params()
        # )
        # ppo.train()
