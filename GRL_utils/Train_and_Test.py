# 该python文件包括对GRL模型的训练以及测试函数
import numpy as np


def Training_GRLModels(GRL_Net, GRL_model, env, n_episodes, max_episode_len, save_dir, warmup, debug):
    """
        该函数为针对GRL模型的训练函数

        参数说明:
        --------
        GRL_Net: GRL模型中采用的神经网络
        GRL_model:需要训练的GRL模型
        env: 注册至gym下的仿真环境
        n_episodes: 训练的回合数
        max_episode_len: 单步训练最大步长
        save_dir: 模型保存路径
        warmup: 模型自由探索步长（随机选择动作）
        debug: 模型参数调试相关
    """
    # 以下为模型训练过程
    Rewards = []  # 初始化奖励矩阵以进行数据保存
    Loss = []  # 初始化Loss矩阵以进行数据保存
    Episode_Steps = []  # 初始化步长矩阵保存每一episode的任务完成时的步长
    Average_Q = []  # 初始化平均Q值矩阵保存每一episode的平均Q值

    # 定义warmup步长
    Warmup_Steps = warmup
    # 定义warmup步长记录
    warmup_count = 0

    for i in range(1, n_episodes + 1):
        # 如果需要调试，则实时打印网络中的参数
        if debug:
            print("------------------------------------")
            for parameters in GRL_Net.parameters():
                print("param:", parameters)
            print("------------------------------------")
        obs = env.reset()
        R = 0  # 行为奖励
        t = 0  # 时间步长
        while True:
            action = GRL_model.act(obs)  # 这里引用了dqn.py中的batch_act函数
            if warmup_count <= Warmup_Steps:  # 进行warmup
                action = np.random.choice(np.arange(3), 40)
            # print("action: ", action)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            warmup_count += 1

            reset = t == max_episode_len
            GRL_model.observe(obs, reward, done, reset)
            if done or reset:
                break
        # 记录训练数据
        Rewards.append(R)  # 记录Rewards
        Episode_Steps.append(t)  # 记录Steps
        # 记录Loss
        Training_Data = GRL_model.get_statistics()
        Loss_episode = Training_Data[1][1]  # 从元组中提取Loss值
        Average_Q_episode = Training_Data[0][1]  # 从元组中提取Average_Q值
        Loss.append(Loss_episode)
        Average_Q.append(Average_Q_episode)
        if i % 1 == 0:
            print('Training Episode:', i, 'Reward:', R)
        if i % 1 == 0:
            print('Statistics:', GRL_model.get_statistics())
    print('Training Finished.')

    # 模型保存
    GRL_model.save(save_dir)
    # 保存训练过程中的各项数据
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Loss", Loss)
    np.save(save_dir + "/Average_Q", Average_Q)


def Testing_GRLModels(GRL_Net, GRL_model, env, test_episodes, load_dir, debug):
    """
        该函数为针对训练好的GRL模型的测试函数

        参数说明:
        --------
        GRL_Net: GRL模型中采用的神经网络
        GRL_model:需要测试的GRL模型
        env: 注册至gym下的仿真环境
        test_episodes: 测试的回合数
        load_dir: 模型读取路径
        debug: 模型参数调试相关
    """
    # 以下为模型测试过程
    Rewards = []  # 初始化奖励矩阵以进行数据保存

    GRL_model.load(load_dir)
    for i in range(1, test_episodes + 1):
        # 如果需要调试，则实时打印网络中的参数
        if debug:
            print("------------------------------------")
            for parameters in GRL_Net.parameters():
                print("param:", parameters)
            print("------------------------------------")
        obs = env.reset()
        R = 0
        t = 0
        while True:
            action = GRL_model.act(obs)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            reset = done
            GRL_model.observe(obs, reward, done, reset)
            if done or reset:
                break
        Rewards.append(R)  # 记录Rewards
        print('Evaluation Episode:', i, 'Reward:', R)
    print('Evaluation Finished')

    # 测试数据保存
    np.save(load_dir + "/Test_Rewards", Rewards)
