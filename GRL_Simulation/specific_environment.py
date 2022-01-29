from environment import Env
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time

from gym.spaces.box import Box
from gym.spaces import Discrete
from gym.spaces import Tuple


class MergeEnv(Env):

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.
        """
        N = self.net_params.additional_params['num_vehicles']
        F = 2 + self.net_params.additional_params['highway_lanes'] \
            + self.n_unique_intentions

        states = Box(low=-np.inf, high=np.inf, shape=(N, F), dtype=np.float32)
        adjacency = Box(low=0, high=1, shape=(N, N), dtype=np.int32)
        mask = Box(low=0, high=1, shape=(N,), dtype=np.int32)

        return Tuple([states, adjacency, mask])

    @property
    def action_space(self):
        N = self.net_params.additional_params['num_vehicles']
        return Box(low=0, high=1, shape=(N,), dtype=np.int32)
        # return Discrete(3)

    def get_state(self):  # 该函数返回构造的节点特征矩阵，稠密邻接矩阵以及mask矩阵
        """construct a graph for each time step
        """
        N = self.net_params.additional_params['num_vehicles']
        # num_cav = self.net_params.additional_params['num_cav'] # maximum number of CAVs
        num_hv = self.net_params.additional_params['num_hv']  # maximum number of HDVs

        num_lanes = self.net_params.additional_params['highway_lanes']

        ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()

        # filter the ones on the ramps
        rl_ids = [id_ for id_ in rl_ids if not self.k.vehicle.get_edge(id_).startswith('off_ramp')]
        rl_ids = sorted(rl_ids)
        # print("rl:", rl_ids)

        human_ids = sorted(self.k.vehicle.get_human_ids())
        # print("human:", human_ids)

        # If too many human ids
        if len(human_ids) > num_hv:
            human_ids = human_ids[:num_hv]

        # assert len(ids) != len(human_ids) + len(rl_ids)

        # 初始化状态空间矩阵，邻接矩阵和mask
        states = np.zeros([N, 2 + num_lanes + self.n_unique_intentions])
        adjacency = np.zeros([N, N])
        mask = np.zeros(N)

        if rl_ids:  ## when there is rl_vehicles in the scenario

            ids = human_ids + rl_ids

            # numerical data (speed, location)
            speeds = np.array(self.k.vehicle.get_speed(ids)).reshape(-1, 1)

            # positions = np.array([self.k.vehicle.get_absolute_position(i) for i in ids])  # x y location
            xs = np.array([self.k.vehicle.get_x_by_id(i) for i in ids]).reshape(-1, 1)

            # categorical data 1 hot encoding: (lane location, intention)
            lanes_column = np.array(self.k.vehicle.get_lane(ids))  # 当前环境中的车辆所在车道的编号
            lanes = np.zeros([len(ids), num_lanes])  # 初始化车道onehot矩阵（当前车辆数量x车道数量）
            lanes[np.arange(len(ids)), lanes_column] = 1  # 根据每辆车当前所处的车道，在矩阵对应位置赋值为1
            # print(np.arange(len(ids)))
            # print("lanes_column", lanes_column)
            # print("lanes:", lanes)

            # intention encoding
            # 获得当前环境中车辆的类别，0为有人车辆，1为匝道1驶出的RL车辆，2为匝道2驶出的RL车辆；且获得的矩阵的元素按0，1，2顺序排列
            types_column = np.array([self.intention_dict[self.k.vehicle.get_type(i)] for i in ids])
            intention = np.zeros([len(ids), self.n_unique_intentions])  # 初始化intention矩阵（当前车辆数量x车辆种类）
            intention[np.arange(len(ids)), types_column] = 1  # 根据当前环境中的车辆的类型为intention矩阵赋值

            observed_states = np.c_[xs, speeds, lanes, intention]  # 将上述相关矩阵按列合成为状态观测矩阵

            # assemble into the NxF states matrix
            # 将上述对环境的观测储存至状态矩阵中
            states[:len(human_ids), :] = observed_states[:len(human_ids), :]
            states[num_hv:num_hv + len(rl_ids), :] = observed_states[len(human_ids):, :]

            states[:, 0] /= self.net_params.additional_params['highway_length']

            # construct the adjacency matrix
            # 生成邻接矩阵
            # 使用sklearn库中的欧几里德距离函数计算环境中两两车辆的水平距离（x坐标，维度当前车辆x当前车辆）
            dist_matrix = euclidean_distances(xs)
            adjacency_small = np.zeros_like(dist_matrix)  # 根据dist_matrix生成维度相同的全零邻接矩阵
            adjacency_small[dist_matrix < 20] = 1
            adjacency_small[-len(rl_ids):, -len(rl_ids):] = 1  # 将RL车辆之间在邻接矩阵中进行赋值

            # assemble into the NxN adjacency matrix
            # 将上述small邻接矩阵储存至稠密邻接矩阵中
            adjacency[:len(human_ids), :len(human_ids)] = adjacency_small[:len(human_ids), :len(human_ids)]
            adjacency[num_hv:num_hv + len(rl_ids), :len(human_ids)] = adjacency_small[len(human_ids):, :len(human_ids)]
            adjacency[:len(human_ids), num_hv:num_hv + len(rl_ids)] = adjacency_small[:len(human_ids), len(human_ids):]
            adjacency[num_hv:num_hv + len(rl_ids), num_hv:num_hv + len(rl_ids)] = adjacency_small[len(human_ids):,
                                                                                  len(human_ids):]

            # construct the mask
            # 构造mask矩阵
            mask[num_hv:num_hv + len(rl_ids)] = np.ones(len(rl_ids))

            self.observed_cavs = rl_ids  # RL车辆
            self.observed_all_vehs = ids  # 全部车辆

        return states, adjacency, mask

    def compute_reward(self, rl_actions, **kwargs):  # 该函数用来计算奖励值
        # w_intention = 10
        w_intention = 3
        w_speed = 0.8
        w_p_lane_change = 0.05
        w_p_crash = 0.8
        # w_p_crash = 0

        unit = 1

        # reward for system speed: mean(speed/max_speed) for every vehicle
        speed_reward = 0
        intention_reward = 0

        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) != 0:  # 若观测到RL车辆
            # all_speed = np.array(self.k.vehicle.get_speed(self.observed_all_vehs))
            # max_speed = np.array([self.env_params.additional_params['max_hv_speed']]*(len(self.observed_all_vehs) - len(self.observed_cavs))\
            #                     +[self.env_params.additional_params['max_cav_speed']]*len(self.observed_cavs))

            # all_speed = np.array(self.k.vehicle.get_speed(self.observed_cavs))
            all_speed = np.array(self.k.vehicle.get_speed(rl_ids))
            max_speed = self.env_params.additional_params['max_av_speed']
            speed_reward = np.mean(all_speed / max_speed)
            # print("cavs:", self.observed_cavs)
            # print("all_speed:", all_speed)
            # print("speed_reward:", speed_reward)

            ###### reward for satisfying intention ---- only a big instant reward
            # intention_reward = kwargs['num_full_filled'] * unit + kwargs['num_half_filled'] * unit * 0.5
            intention_reward = self.compute_intention_rewards()  # 计算意图奖励

        # penalty for frequent lane changing behavors
        # 这部分计算对频繁换道的处罚（负奖励）
        drastic_lane_change_penalty = 0
        if self.drastic_veh_id:
            drastic_lane_change_penalty += len(self.drastic_veh_id) * unit

        # penalty for crashing
        # 对于碰撞的惩罚（负奖励）
        total_crash_penalty = 0
        crash_ids = kwargs["fail"]
        # print("kwargs: ", kwargs)
        # print("crash:", crash_ids)
        total_crash_penalty = crash_ids * unit
        # print("total_crash_penalty:", total_crash_penalty)
        # if crash_ids:
        #     print(crash_ids,total_crash_penalty)

        # print(speed_reward, intention_reward, total_crash_penalty, drastic_lane_change_penalty)
        # 这里计算奖励时的系数可能存在问题
        return w_speed * speed_reward + \
               w_intention * intention_reward - \
               w_p_lane_change * drastic_lane_change_penalty - \
               w_p_crash * total_crash_penalty

    def compute_intention_rewards(self):  # 该函数用来计算intention（意图）奖励值

        intention_reward = 0
        try:
            for cav_id in self.observed_cavs:
                cav_lane = self.k.vehicle.get_lane(cav_id)
                cav_edge = self.k.vehicle.get_edge(cav_id)
                cav_type = self.k.vehicle.get_type(cav_id)
                # print("cav_lane:", cav_lane, "\ncav_edge:", cav_edge, "\ncav_type:", cav_type)

                x = self.k.vehicle.get_x_by_id(cav_id)

                if cav_type == "merge_0":
                    if cav_edge == 'highway_0':
                        val = (self.net_params.additional_params['off_ramps_pos'][0] - x) / \
                              self.net_params.additional_params['off_ramps_pos'][0]
                        if cav_lane == 0:
                            intention_reward += val
                        elif cav_lane == 2:
                            intention_reward -= (1 - val)

                elif cav_type == "merge_1":

                    if cav_edge == "highway_0" and cav_lane == 0:
                        val = (self.net_params.additional_params['off_ramps_pos'][0] - x) / \
                              self.net_params.additional_params['off_ramps_pos'][0]
                        intention_reward += val - 1


                    elif cav_edge == "highway_1":
                        val = (self.net_params.additional_params['off_ramps_pos'][1] - x) / (
                                self.net_params.additional_params['off_ramps_pos'][1] -
                                self.net_params.additional_params['off_ramps_pos'][0])
                        if cav_lane == 0:
                            intention_reward += val
                        elif cav_lane == 2:
                            intention_reward -= (1 - val)

                    else:
                        pass
                else:
                    raise Exception("unknow cav type")
        except:
            pass

            # print(cav_id,x,cav_lane)
            # if cav_lane == 0:
            #     # print('here')
            #     x = self.k.vehicle.get_x_by_id(cav_id)
            #     cav_edge = self.k.vehicle.get_edge(cav_id)
            #     cav_type = self.k.vehicle.get_type(cav_id)
            #     # total_length = self.net_params.additional_params['highway_length']
            #     if (cav_type == 'merge_0' and cav_edge == 'highway_0'):
            #         val = (self.net_params.additional_params['off_ramps_pos'][0] - x)/self.net_params.additional_params['off_ramps_pos'][0]
            #         intention_reward += val
            #         # print('1: ',cav_id,val)
            #     elif (cav_type == 'merge_1' and cav_edge == 'highway_1'):
            #         val = (self.net_params.additional_params['off_ramps_pos'][1] - x)/(self.net_params.additional_params['off_ramps_pos'][1] - self.net_params.additional_params['off_ramps_pos'][0])
            #         intention_reward += val
            #         # print('2: ', cav_id, val)
            #     elif (cav_type == 'merge_1' and cav_edge == 'highway_0'):
            #         val = (self.net_params.additional_params['off_ramps_pos'][0] - x)/self.net_params.additional_params['off_ramps_pos'][0]
            #         intention_reward -= (1-val)
            # print('3: ', cav_id, (1-val))

        return intention_reward

    def apply_rl_actions(self, rl_actions=None):  # 执行强化学习的行为
        ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        if isinstance(rl_actions, np.ndarray):
            # rl_actions = rl_actions.reshape((self.net_params.additional_params['num_cav'],3))
            rl_actions2 = rl_actions.copy()
            rl_actions2 -= 1
            # rl_ids = self.observed_cavs
            drastic_veh = []
            for ind, veh_id in enumerate(rl_ids):  # 这部分通过计算当前时间以及最后一次换道的时间间隔来检测车辆是否有激烈换到行为
                if rl_actions2[ind] != 0 and (self.time_counter - self.k.vehicle.get_last_lc(veh_id) < 50):
                    drastic_veh.append(veh_id)
                    # print("drastic lane change: ", veh_id)

            self.drastic_veh_id = drastic_veh
            # if len(rl_ids) != 0:  # GCQ只针对RL车辆进行控制，故若空间中没有RL车辆，则跳过控制环节
            #     self.k.vehicle.apply_lane_change(rl_ids, rl_actions2)
            # else:
            #     pass

        return None

    def check_full_fill(self):  # 统计成功从对应匝道驶出的RL车辆
        rl_veh_ids = self.k.vehicle.get_rl_ids()
        num_full_filled = 0
        num_half_filled = 0
        for rl_id in rl_veh_ids:
            if rl_id not in self.exited_vehicles:
                current_edge = self.k.vehicle.get_edge(rl_id)
                if current_edge in self.terminal_edges:
                    self.exited_vehicles.append(rl_id)
                    veh_type = self.k.vehicle.get_type(rl_id)

                    # check if satisfy the intention

                    if self.n_unique_intentions == 3:  # specific merge
                        if (veh_type == 'merge_0' and current_edge == 'off_ramp_0') \
                                or (veh_type == 'merge_1' and current_edge == 'off_ramp_1'):
                            num_full_filled += 1
                            print('satisfied: ', rl_id)

                    elif self.n_unique_intentions == 2:  # nearest merge
                        num_full_filled += (current_edge == 'off_ramp_0') * 1
                        num_half_filled += (current_edge == 'off_ramp_1') * 1
                        print("wrongs")

                    else:
                        raise Exception("unknown num of unique n_unique_intentions")
        return num_full_filled, num_half_filled
