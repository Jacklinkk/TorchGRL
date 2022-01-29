"""Contains an experiment class for running simulations."""
from flow.core.util import emission_to_csv
from registry_custom import make_create_env
import datetime
import logging
import time
import os
import numpy as np
import json


class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, training, testing, num_human, actual_num_human, num_cav, model, debug, num_merge_0=None,
            num_merge_1=None):

        # 设置模型名称
        model_name = model + '_hv_' + str(num_human) + '_cav_' + str(num_cav)

        # F为特征长度，N为智能体数量，A为可采用的action的数量
        F = 2 + self.env.net_params.additional_params[
            'highway_lanes'] + self.env.n_unique_intentions  # input feature size
        N = num_human + num_cav
        A = 3
        # 是否采用新训练的模型进行测试

        # 导入强化学习gym相关库
        from gym.spaces.box import Box
        from gym.spaces import Discrete
        from gym.spaces.dict import Dict

        # states为状态矩阵，adjacency为邻接矩阵（对应图神经网络部分），mask为掩膜矩阵（起智能体过滤作用）
        states = Box(low=-np.inf, high=np.inf, shape=(N, F), dtype=np.float32)
        adjacency = Box(low=0, high=1, shape=(N, N), dtype=np.int32)
        mask = Box(low=0, high=1, shape=(N,), dtype=np.int32)

        # obs_space为状态观测矩阵，act_space为动作空间矩阵
        obs_space = Dict({'states': states, 'adjacency': adjacency, 'mask': mask})
        act_space = Box(low=0, high=1, shape=(N,), dtype=np.int32)

        # 初始化DQN类
        import pfrl
        import torch
        import torch.nn
        from GRLNet.Pytorch_GRL_Dueling import torch_GRL_Deuling  # 导入编写的pytorch下的网络
        from GRL_utils.Train_and_Test import Training_GRLModels, Testing_GRLModels  # 导入自行编写的相关工具

        # 初始化GRL网络
        GRL = torch_GRL_Deuling(N, F, obs_space, act_space, A)
        # 初始化优化器
        optimizer = torch.optim.Adam(GRL.parameters(), eps=0.0001)
        # 定义数据缓冲器
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
        # 定义折扣因子
        gamma = 0.99
        # 定义智能体策略参数
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=self.env.action_space.sample)
        # 定义计算设备(cuda:0)
        gpu = 0
        # 定义特征提取器(转换成float32类型保证pytorch可以接收特征)
        phi = lambda x: x.astype(np.float32, copy=False)

        # 初始化DoubleDQN类
        warmup = 20000
        GRL_DuelingDoubleDqn = pfrl.agents.DoubleDQN(
            GRL,  # 模型采用的网络
            optimizer,  # 模型采用的优化器
            replay_buffer,
            gamma,
            explorer,
            minibatch_size=32,
            replay_start_size=warmup,
            update_interval=10,
            target_update_interval=1000,
            target_update_method='soft',
            soft_update_tau=0.01,
            phi=phi,
            gpu=gpu,
        )

        # 进行模型训练
        n_episodes = 150
        max_episode_len = 2500
        save_dir = 'GRL_Trained_Models/DD_DQN'
        debug_training = False
        if training:
            Training_GRLModels(GRL, GRL_DuelingDoubleDqn, self.env, n_episodes, max_episode_len, save_dir, warmup,
                               debug_training)

        # 进行模型测试
        test_episodes = 10
        load_dir = 'Test_Models/DD_DQN/DD_DQN_3'
        debug_testing = False
        if testing:
            Testing_GRLModels(GRL, GRL_DuelingDoubleDqn, self.env, test_episodes, load_dir, debug_testing)

            # 设置logger以打印训练信息以便于理解
            # import logging
            # import sys
            # logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
            #
            # pfrl.experiments.train_agent_with_evaluation(
            #     GRL_dqn,
            #     self.env,
            #     steps=25000,  # Train the agent for 2000 steps
            #     eval_n_steps=None,  # We evaluate for episodes, not time
            #     eval_n_episodes=10,  # num episodes are sampled for each evaluation
            #     train_max_episode_len=2500,  # Maximum length of each episode
            #     eval_interval=5000,  # Evaluate the agent after every 1000 steps
            #     outdir='result',  # Save everything to 'result' directory
            # )
