# 该python文件用来绘制结果曲线
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def Data_Loader(data_dir):
    """
        此函数用来读取保存的数据

        参数说明:
        --------
        data_dir: 模型及数据保存目录
    """

    # 获取目录
    Reward_dir = data_dir + "/Test_Rewards.npy"

    # 通过numpy读取数据
    Reward = np.load(Reward_dir)

    return Reward


def Mean_and_Std(Data):
    """
        此函数用来计算不同sample下数据的平均值以及标准差

        参数说明:
        --------
        Data: 需要计算的数据list.
            Data中每个list的数据形式为[Reward, Loss, Average_Q, Episode]
    """

    # 获取数据list的长度
    Length_Data = len(Data)

    # 计算各个指标的平均值以及标准差
    # -------------------------------------------------------------- #
    # 1.针对Reward进行处理
    Reward = []  # 构建初始矩阵
    for i in range(0, Length_Data):
        Reward.append(Data[i][0])
    Reward_Average = np.average(Reward, axis=0)  # 按列计算每个step的均值
    Reward_Std = np.std(Reward, axis=0)  # 按列计算每个step的标准差
    Reward_Proceed = [Reward_Average, Reward_Std]  # 将数据组合成矩阵
    # -------------------------------------------------------------- #


if __name__ == '__main__':
    # (1) 数据处理(3 samples)
    # 1.目录输入
    DQN_dir1 = "Logging_Testing/DQN/DQN_1"
    DQN_dir2 = "Logging_Testing/DQN/DQN_2"
    DQN_dir3 = "Logging_Testing/DQN/DQN_3"

    DoubleDQN_dir1 = "Logging_Testing/DoubleDQN/DoubleDQN_1"
    DoubleDQN_dir2 = "Logging_Testing/DoubleDQN/DoubleDQN_2"
    DoubleDQN_dir3 = "Logging_Testing/DoubleDQN/DoubleDQN_3"

    DuelingDQN_dir1 = "Logging_Testing/DuelingDQN/DuelingDQN_1"
    DuelingDQN_dir2 = "Logging_Testing/DuelingDQN/DuelingDQN_2"
    DuelingDQN_dir3 = "Logging_Testing/DuelingDQN/DuelingDQN_3"

    DD_DQN_dir1 = "Logging_Testing/DD_DQN/DD_DQN_1"
    DD_DQN_dir2 = "Logging_Testing/DD_DQN/DD_DQN_2"
    DD_DQN_dir3 = "Logging_Testing/DD_DQN/DD_DQN_3"

    # 2.数据读取
    Data_DQN1 = Data_Loader(DQN_dir1)
    Data_DQN2 = Data_Loader(DQN_dir2)
    Data_DQN3 = Data_Loader(DQN_dir3)
    Data_DQN = [Data_DQN1, Data_DQN2, Data_DQN3]

    Data_DoubleDQN1 = Data_Loader(DoubleDQN_dir1)
    Data_DoubleDQN2 = Data_Loader(DoubleDQN_dir2)
    Data_DoubleDQN3 = Data_Loader(DoubleDQN_dir3)
    Data_DoubleDQN = [Data_DoubleDQN1, Data_DoubleDQN2, Data_DoubleDQN3]

    Data_DuelingDQN1 = Data_Loader(DuelingDQN_dir1)
    Data_DuelingDQN2 = Data_Loader(DuelingDQN_dir2)
    Data_DuelingDQN3 = Data_Loader(DuelingDQN_dir3)
    Data_DuelingDQN = [Data_DuelingDQN1, Data_DuelingDQN2, Data_DuelingDQN3]

    Data_DD_DQN1 = Data_Loader(DD_DQN_dir1)
    Data_DD_DQN2 = Data_Loader(DD_DQN_dir2)
    Data_DD_DQN3 = Data_Loader(DD_DQN_dir3)
    Data_DD_DQN = [Data_DD_DQN1, Data_DD_DQN2, Data_DD_DQN3]

    # 3.数据均值计算
    DQN_Mean = np.mean(Data_DQN)
    DoubleDQN_Mean = np.mean(Data_DoubleDQN)
    DuelingDQN_Mean = np.mean(Data_DuelingDQN)
    DD_DQN_Mean = np.mean(Data_DD_DQN)

    # 4.打印平均奖励
    print("------------------ Reward ------------------")
    print("DQN_Reward:", DQN_Mean)
    print("DoubleDQN_Reward:", DoubleDQN_Mean)
    print("DuelingDQN_Reward:", DuelingDQN_Mean)
    print("DD_DQN_Reward:", DD_DQN_Mean)


