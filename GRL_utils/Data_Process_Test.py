# This python file is used to plot the training curve
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def Data_Loader(data_dir):
     """
        This function is used to load the training data
        Parameter Description:
        --------
        data_dir: model and data storage directory
    """

    # get directory
    Reward_dir = data_dir + "/Test_Rewards.npy"

    # load data via numpy
    Reward = np.load(Reward_dir)

    return Reward


def Mean_and_Std(Data):
    """
        This function is used to calculate the mean and standard deviation of the data under different samples
        Parameter Description:
        --------
        Data: A list of data to be calculated.
            The data form of each list can be described as [Reward, Loss, Average_Q, Episode]
    """

    # Get the length of the data list
    Length_Data = len(Data)

    # Calculate the mean and standard deviation of each indicator
    # -------------------------------------------------------------- #
    # 1.Reward
    Reward = []   
    for i in range(0, Length_Data):
        Reward.append(Data[i][0])
    Reward_Average = np.average(Reward, axis=0)  
    Reward_Std = np.std(Reward, axis=0)  
    Reward_Proceed = [Reward_Average, Reward_Std] 
    # -------------------------------------------------------------- #


if __name__ == '__main__':
    # (1) data processing (3 samples)
    # 1.directory
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

    # 2.load data
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

    # 3.Calculate the mean reward
    DQN_Mean = np.mean(Data_DQN)
    DoubleDQN_Mean = np.mean(Data_DoubleDQN)
    DuelingDQN_Mean = np.mean(Data_DuelingDQN)
    DD_DQN_Mean = np.mean(Data_DD_DQN)

    # 4.Print the mean reward
    print("------------------ Reward ------------------")
    print("DQN_Reward:", DQN_Mean)
    print("DoubleDQN_Reward:", DoubleDQN_Mean)
    print("DuelingDQN_Reward:", DuelingDQN_Mean)
    print("DD_DQN_Reward:", DD_DQN_Mean)


