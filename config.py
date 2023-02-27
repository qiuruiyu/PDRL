from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class Params:
    q: int = 2
    r: int = 1 
    seed: int = 1234

    num_workers: int = 10
    num_episode: int = 200
    batch_size: int = 2048
    mini_batch_size: int = 256
    num_epoch: int = 10
    num_test: int = 10

    lr_a: float = 5e-4
    lr_c: float = 5e-4
    gamma: float = 0.99
    gae_lam: float = 0.97
    clipped_value_loss_param: Union[float, None] = None
    loss_value_coef: float = 0.5
    loss_entropy_coef: float = 0.01
    target_kl: Union[float, None] = None
    clip: float = 0.2
    max_grad_norm: float = 0.5
    EPS = 1e-5

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Here, use the percentage to represent the noise gotten into the state. e.g. 0.05 <--> 5% noise in state  
    adversary_attack: bool = False
    adversary_attack_var: Union[float, None] = 0
    loss_upper_kl_coef: float = 0.05

    early_stop: bool = True
    early_stop_episode: int = 9999

    model_path = './model/'
    net_structure_log = False
    net_structure_logger_path = './network/'
    tensorboard_log = False
    tensorboard_logger_path = './paper_pic_res_log/'
    save_data = True
    data_path = './paper_pic_res/'
