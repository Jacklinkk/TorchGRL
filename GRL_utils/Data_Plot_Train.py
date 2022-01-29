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
    Reward_dir = data_dir + "/Rewards.npy"
    Episode_dir = data_dir + "/Episode_Steps.npy"
    Loss_dir = data_dir + "/Loss.npy"
    Q_dir = data_dir + "/Average_Q.npy"

    # 通过numpy读取数据
    Reward = np.load(Reward_dir)
    Episode = np.load(Episode_dir)
    Loss = np.load(Loss_dir)
    Average_Q = np.load(Q_dir)

    return [Reward, Loss, Average_Q, Episode]
    # return [Reward, Loss, Episode]


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

    # -------------------------------------------------------------- #
    # 2.针对Loss进行处理
    Loss = []  # 构建初始矩阵
    for i in range(0, Length_Data):
        Loss.append(Data[i][1])
    Loss_Average = np.average(Loss, axis=0)  # 按列计算每个step的均值
    Loss_Std = np.std(Loss, axis=0)  # 按列计算每个step的标准差
    Loss_Proceed = [Loss_Average, Loss_Std]  # 将数据组合成矩阵
    # -------------------------------------------------------------- #

    # -------------------------------------------------------------- #
    # 3.针对Average_Q进行处理
    Average_Q = []  # 构建初始矩阵
    for i in range(0, Length_Data):
        Average_Q.append(Data[i][2])
    Average_Q_Average = np.average(Average_Q, axis=0)  # 按列计算每个step的均值
    Average_Q_Std = np.std(Loss, axis=0)  # 按列计算每个step的标准差
    Average_Q_Proceed = [Average_Q_Average, Average_Q_Std]  # 将数据组合成矩阵
    # -------------------------------------------------------------- #

    return [Reward_Proceed, Loss_Proceed, Average_Q_Proceed]


if __name__ == '__main__':
    # (1) 数据处理(3 samples)
    # 1.目录输入
    DQN_dir1 = "Logging_Training/DQN/DQN_1"
    DQN_dir2 = "Logging_Training/DQN/DQN_2"
    DQN_dir3 = "Logging_Training/DQN/DQN_3"

    DoubleDQN_dir1 = "Logging_Training/DoubleDQN/DoubleDQN_1"
    DoubleDQN_dir2 = "Logging_Training/DoubleDQN/DoubleDQN_2"
    DoubleDQN_dir3 = "Logging_Training/DoubleDQN/DoubleDQN_3"

    DuelingDQN_dir1 = "Logging_Training/DuelingDQN/DuelingDQN_1"
    DuelingDQN_dir2 = "Logging_Training/DuelingDQN/DuelingDQN_2"
    DuelingDQN_dir3 = "Logging_Training/DuelingDQN/DuelingDQN_3"

    DD_DQN_dir1 = "Logging_Training/DD_DQN/DD_DQN_1"
    DD_DQN_dir2 = "Logging_Training/DD_DQN/DD_DQN_2"
    DD_DQN_dir3 = "Logging_Training/DD_DQN/DD_DQN_3"

    Rule_dir1 = "Logging_Training/Rule_Based/Rule_Based1"
    Rule_dir2 = "Logging_Training/Rule_Based/Rule_Based2"
    Rule_dir3 = "Logging_Training/Rule_Based/Rule_Based3"

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

    Data_Rule_Based1 = Data_Loader(Rule_dir1)
    Data_Rule_Based2 = Data_Loader(Rule_dir2)
    Data_Rule_Based3 = Data_Loader(Rule_dir3)
    Data_Rule_Based = [Data_Rule_Based1, Data_Rule_Based2, Data_Rule_Based3]

    # 3.数据均值和标准差计算
    Data_DQN = Mean_and_Std(Data_DQN)
    Data_DoubleDQN = Mean_and_Std(Data_DoubleDQN)
    Data_DuelingDQN = Mean_and_Std(Data_DuelingDQN)
    Data_DD_DQN = Mean_and_Std(Data_DD_DQN)
    Data_Rule_Based = Mean_and_Std(Data_Rule_Based)

    # (2) 绘制图像
    # ------------------------------------ #
    # 1.绘制奖励曲线
    fig_Reward, ax_Reward = plt.subplots(dpi=240)

    # 定义横轴(episode)
    length = len(Data_DQN[0][0])
    x = np.arange(0, length, 1)

    # 定义纵轴(奖励)
    DQN_Reward = Data_DQN[0][0]
    DoubleDQN_Reward = Data_DoubleDQN[0][0]
    DuelingDQN_Reward = Data_DuelingDQN[0][0]
    DD_DQN_Reward = Data_DD_DQN[0][0]
    Rule_Based_Reward = Data_Rule_Based[0][0]

    # 计算并打印平均奖励
    print("------------------ Reward ------------------")
    print("Rule_Based_Reward:", np.mean(Rule_Based_Reward[16:]))
    print("DQN_Reward:", np.mean(DQN_Reward[16:]))
    print("DoubleDQN_Reward:", np.mean(DoubleDQN_Reward[16:]))
    print("DuelingDQN_Reward:", np.mean(DuelingDQN_Reward[16:]))
    print("DD_DQN_Reward:", np.mean(DD_DQN_Reward[16:]))

    # 曲线平滑处理
    sigma = 1.0
    Rule_Based_Reward = gaussian_filter1d(Rule_Based_Reward, sigma=sigma)
    DQN_Reward = gaussian_filter1d(DQN_Reward, sigma=sigma)
    DoubleDQN_Reward = gaussian_filter1d(DoubleDQN_Reward, sigma=sigma)
    DuelingDQN_Reward = gaussian_filter1d(DuelingDQN_Reward, sigma=sigma)
    DD_DQN_Reward = gaussian_filter1d(DD_DQN_Reward, sigma=sigma)

    # 曲线绘制
    # ax_Reward.fill_between(x, DQN_Reward+Data_DQN[0][1], DQN_Reward-Data_DQN[0][1])
    linewidth = 1.2
    fontsize = 11.5
    ax_Reward.plot(x, Rule_Based_Reward, '-k', linewidth=linewidth, label='Rule-Based')
    ax_Reward.plot(x, DQN_Reward, 'b', linewidth=linewidth, label='DQN')
    ax_Reward.plot(x, DoubleDQN_Reward, 'r', linewidth=linewidth, label='Double DQN')
    ax_Reward.plot(x, DuelingDQN_Reward, 'y', linewidth=linewidth, label='Dueling DQN')
    ax_Reward.plot(x, DD_DQN_Reward, 'g', linewidth=linewidth, label='DD-DQN')
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    ax_Reward.set_xlabel("Episode", size=fontsize)
    ax_Reward.set_ylabel("Reward", size=fontsize)
    ax_Reward.set_xlim([1, length])
    # ax_Reward.set_ylim([-500, 4000])
    ax_Reward.grid(True)
    ax_Reward.legend(loc='center right', bbox_to_anchor=(1, 0.65))

    # 保存曲线
    plt.savefig(fname="Fig/Fig_Training/Reward.jpg", dpi='figure')
    plt.show()
    # ------------------------------------ #

    # ------------------------------------ #
    # 2.绘制Loss曲线
    fig_Loss, ax_Loss = plt.subplots(dpi=240)

    # 定义横轴(episode)
    length = len(Data_DQN[1][0])
    x = np.arange(0, length, 1)

    # 定义纵轴(平均Loss)
    DQN_Loss = Data_DQN[1][0]
    DoubleDQN_Loss = Data_DoubleDQN[1][0]
    DuelingDQN_Loss = Data_DuelingDQN[1][0]
    DD_DQN_Loss = Data_DD_DQN[1][0]

    # 计算并打印平均Loss
    print("------------------ Loss ------------------")
    print("DQN_Loss:", np.mean(DQN_Loss[13:]))
    print("DoubleDQN_Loss:", np.mean(DoubleDQN_Loss[13:]))
    print("DuelingDQN_Loss:", np.mean(DuelingDQN_Loss[13:]))
    print("DD_DQN_Loss:", np.mean(DD_DQN_Loss[14:]))

    # 曲线平滑处理
    sigma = 0.5
    DQN_Loss = gaussian_filter1d(DQN_Loss, sigma=sigma)
    DoubleDQN_Loss = gaussian_filter1d(DoubleDQN_Loss, sigma=sigma)
    DuelingDQN_Loss = gaussian_filter1d(DuelingDQN_Loss, sigma=sigma)
    DD_DQN_Loss = gaussian_filter1d(DD_DQN_Loss, sigma=sigma)

    # 曲线绘制
    linewidth = 1.2
    fontsize = 11.5
    ax_Loss.plot(x, DQN_Loss, 'b', linewidth=linewidth, label='DQN')
    ax_Loss.plot(x, DoubleDQN_Loss, 'r', linewidth=linewidth, label='Double DQN')
    ax_Loss.plot(x, DuelingDQN_Loss, 'y', linewidth=linewidth, label='Dueling DQN')
    ax_Loss.plot(x, DD_DQN_Loss, 'g', linewidth=linewidth, label='DD-DQN')
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    ax_Loss.set_xlabel("Episode", size=fontsize)
    ax_Loss.set_ylabel("Average_Loss", size=fontsize)
    ax_Loss.set_xlim([0, length])
    # ax_Loss.set_ylim([-500, 4000])
    ax_Loss.grid(True)
    ax_Loss.legend()

    # 保存曲线
    plt.savefig(fname="Fig/Fig_Training/Loss.jpg", dpi='figure')
    plt.show()
    # ------------------------------------ #

    # ------------------------------------ #
    # 3.绘制Average_Q曲线
    fig_Average_Q, ax_Average_Q = plt.subplots(dpi=240)

    # 定义横轴(episode)
    length = len(Data_DQN[2][0])
    x = np.arange(0, length, 1)

    # 定义纵轴(奖励)
    DQN_Average_Q = Data_DQN[2][0]
    DoubleDQN_Average_Q = Data_DoubleDQN[2][0]
    DuelingDQN_Average_Q = Data_DuelingDQN[2][0]
    DD_DQN_Average_Q = Data_DD_DQN[2][0]

    # 曲线平滑处理
    sigma = 0.8
    DQN_Average_Q = gaussian_filter1d(DQN_Average_Q, sigma=sigma)
    DoubleDQN_Average_Q = gaussian_filter1d(DoubleDQN_Average_Q, sigma=sigma)
    DuelingDQN_Average_Q = gaussian_filter1d(DuelingDQN_Average_Q, sigma=sigma)
    DD_DQN_Average_Q = gaussian_filter1d(DD_DQN_Average_Q, sigma=sigma)
    # 曲线绘制
    linewidth = 1.2
    fontsize = 11.5
    ax_Average_Q.plot(x, DQN_Average_Q, 'b', linewidth=linewidth, label='DQN')
    ax_Average_Q.plot(x, DoubleDQN_Average_Q, 'r', linewidth=linewidth, label='Double DQN')
    ax_Average_Q.plot(x, DuelingDQN_Average_Q, 'y', linewidth=linewidth, label='Dueling DQN')
    ax_Average_Q.plot(x, DD_DQN_Average_Q, 'g', linewidth=linewidth, label='DD-DQN')
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    ax_Average_Q.set_xlabel("Episode", size=fontsize)
    ax_Average_Q.set_ylabel("Average_Q", size=fontsize)
    ax_Average_Q.set_xlim([1, length])
    # ax_Average_Q.set_ylim([-500, 4000])
    ax_Average_Q.grid(True)
    ax_Average_Q.legend()
    # 保存曲线
    plt.savefig(fname="Fig/Fig_Training/Average_Q.jpg", dpi='figure')
    plt.show()
    # ------------------------------------ #

    # # ------------------------------------ #
    # # 4.绘制Episode曲线(这个指标其实可以直接比较平均值)
    # fig_Episode, ax_Episode = plt.subplots(dpi=240)
    #
    # # 定义横轴(episode)
    # length = len(Data_DQN[0])
    # x = np.arange(0, length, 1)
    #
    # # 定义纵轴(奖励)
    # DQN_Episode = Data_DQN[3]
    # DoubleDQN_Episode = Data_DoubleDQN[3]
    #
    # # 曲线绘制
    # ax_Episode.plot(x, DQN_Episode, 'b', linewidth=0.8, label='DQN')
    # ax_Episode.plot(x, DoubleDQN_Episode, 'r', linewidth=0.8, label='DoubleDQN')
    # ax_Episode.set_xlabel("Episode")
    # ax_Episode.set_ylabel("Episode_Steps")
    # ax_Episode.set_xlim([1, length])
    # # ax_Average_Q.set_ylim([-500, 4000])
    # ax_Episode.grid(True)
    # ax_Episode.legend()
    # plt.show()
    # # ------------------------------------ #
