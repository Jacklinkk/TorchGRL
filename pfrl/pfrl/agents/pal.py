import torch

from pfrl.agents import dqn
from pfrl.utils.recurrent import pack_and_forward


class PAL(dqn.DQN):
    """Persistent Advantage Learning.

    See: http://arxiv.org/abs/1512.04860.

    Args:
      alpha (float): Weight of (persistent) advantages. Convergence
        is guaranteed only for alpha in [0, 1).

    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop("alpha", 0.9)
        super().__init__(*args, **kwargs)

    def _compute_y_and_t(self, exp_batch):

        batch_state = exp_batch["state"]
        batch_size = len(exp_batch["reward"])

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model,
                batch_state,
                exp_batch["recurrent_state"],
            )
        else:
            qout = [self.model(elem[0], elem[1], elem[2]) for elem in batch_state]

        batch_actions = exp_batch["action"]
        # 按照DQN程序的改动对以下Q值的计算进行相同改动
        batch_q = []  # 初始化batch_q矩阵
        count = 0  # 定义针对batch_actions的计数器
        for q_eval in qout:
            q_e = q_eval.evaluate_actions(batch_actions[count])  # 计算batch编号为count的q值
            count += 1
            batch_q.append(q_e)
        batch_q = torch.stack(batch_q)  # 将上述循环生成的list转换为tensor数据类型
        batch_q = torch.mean(batch_q, dim=1)

        # Compute target values
        batch_next_state = exp_batch["next_state"]
        with torch.no_grad():
            if self.recurrent:
                target_qout, _ = pack_and_forward(
                    self.target_model,
                    batch_state,
                    exp_batch["recurrent_state"],
                )
                target_next_qout, _ = pack_and_forward(
                    self.target_model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                target_qout = [self.target_model(elem[0], elem[1], elem[2]) for elem in batch_state]
                target_next_qout = [self.target_model(elem[0], elem[1], elem[2]) for elem in batch_next_state]

            # 同样需要对target_q值进行与q值相同方法的处理，以实现与batch_size维度的匹配
            next_q_max = [elem.max for elem in target_next_qout]
            next_q_max = torch.stack(next_q_max)
            next_q_max = torch.mean(next_q_max, dim=1)
            # next_q_max = torch.reshape(target_next_qout.max, (batch_size,))

            batch_rewards = exp_batch["reward"]
            batch_terminal = exp_batch["is_state_terminal"]

            # T Q: Bellman operator
            t_q = (
                batch_rewards
                + exp_batch["discount"] * (1.0 - batch_terminal) * next_q_max
            )

            # T_PAL Q: persistent advantage learning operator
            # 由于原始程序对多智能体强化学习的不兼容，同样需要进行程序改动
            cur_advantage=[]  # 初始化当前优势矩阵
            next_advantage=[]  # 初始化next_step优势矩阵
            count = 0  # # 定义针对batch_actions的计数器
            for qt_cur, qt_next in zip(target_qout, target_next_qout):  # 遍历多个Q值列表
                qt_cur_advantage = qt_cur.compute_advantage(batch_actions[count])
                qt_next_advantage = qt_next.compute_advantage(batch_actions[count])
                count += 1
                cur_advantage.append(qt_cur_advantage)
                next_advantage.append(qt_next_advantage)
            cur_advantage = torch.stack(cur_advantage)
            cur_advantage = torch.mean(cur_advantage, dim=1)
            next_advantage = torch.stack(next_advantage)
            next_advantage = torch.mean(next_advantage, dim=1)

            # cur_advantage = torch.reshape(
            #     target_qout.compute_advantage(batch_actions), (batch_size,)
            # )
            # next_advantage = torch.reshape(
            #     target_next_qout.compute_advantage(batch_actions), (batch_size,)
            # )
            tpal_q = t_q + self.alpha * torch.max(cur_advantage, next_advantage)

        return batch_q, tpal_q
