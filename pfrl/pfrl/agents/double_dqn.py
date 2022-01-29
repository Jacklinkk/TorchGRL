import torch
from pfrl.agents import dqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward


class DoubleDQN(dqn.DQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch["next_state"]

        with evaluating(self.model):
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = [self.model(elem[0], elem[1], elem[2]) for elem in batch_next_state]

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = [self.target_model(elem[0], elem[1], elem[2]) for elem in batch_next_state]

        # print("next_qout:", next_qout)
        # print("target_next_qout:", target_next_qout)

        # 计算next_q_max时，要分别针对每个batch的q值进行评估
        next_q_max = []  # 初始化next_q_max矩阵
        count = 0  # 定义针对next_qout中，greedy_actions的计数器
        for q_eval in target_next_qout:
            q_e = q_eval.evaluate_actions(next_qout[count].greedy_actions)
            count += 1
            next_q_max.append(q_e)
        next_q_max = torch.stack(next_q_max)
        next_q_max = torch.mean(next_q_max, dim=1)
        # next_q_max = target_next_qout.evaluate_actions(next_qout.greedy_actions)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max
