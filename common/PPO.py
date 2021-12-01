import torch
from net_builder import Common, Actor_builder, Critic_builder
from torch.distributions import Normal
from itertools import chain
from collections import deque
import numpy as np
import copy

LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4

class PPO:
    def __init__(self, in_channel, in_shape, action_space, batch_size):
        self.action_space = action_space
        self.input_channel = in_channel
        self.pixel_shape = in_shape
        self.batch_size = batch_size
        self._init(self.input_channel, self.batch_size)
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.decay_index = 0.95
        self.epilson = 0.2
        self.c_loss = torch.nn.MSELoss()
        self.c_opt = torch.optim.Adam(params=chain(self.common.parameters(), self.v.parameters()), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=chain(self.common.parameters(), self.pi.parameters()), lr=self.lr_actor)
        self.update_actor_epoch = 5
        self.update_critic_epoch = 5
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, in_channel, train_batch):
        self.common = Common(in_channel)
        self.pi = Actor_builder()
        self.commonold = Common(in_channel)
        self.piold = Actor_builder()
        self.v = Critic_builder()
        self.memory = deque(maxlen=train_batch)

    def get_action(self, obs_, speed_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        speed_ = torch.Tensor(copy.deepcopy(speed_))
        self.common.eval()
        self.pi.eval()
        common_feature = self.common(obs_, speed_)
        (acc_m, acc_s), (ori_m, ori_s) = self.pi(common_feature)
        # print(f'mu: {mean.cpu().item()}')
        dist = Normal(ori_m.cpu().detach(), ori_s.cpu().detach())
        accel = Normal(acc_m.cpu().detach(), acc_s.cpu().detach())

        prob_ori = dist.sample()
        log_prob_ori = dist.log_prob(prob_ori)
        prob_accel = accel.sample()
        log_prob_accel = accel.log_prob(prob_accel)

        self.common.train()
        self.pi.train()

        return (prob_accel, log_prob_accel), (prob_ori, log_prob_ori)

    def state_store_memory(self, s, acc, ori, r, logprob_acc, logprob_ori):
        self.memory.append((s, acc, ori, r, logprob_acc, logprob_ori))

    # 计算reward衰减，根据马尔可夫过程，从最后一个reward向前推
    def decayed_reward(self, singal_state_frame, singal_speed_frame, reward_):
        decayed_rd = []
        state_frame = torch.Tensor(singal_state_frame)
        speed_frame = torch.Tensor(singal_speed_frame)
        common_feature = self.common(state_frame, speed_frame)
        value_target = self.v(common_feature).detach().numpy()
        for rd_ in reward_[::-1]:
            value_target = rd_ + value_target * self.decay_index
            decayed_rd.append(value_target)
        decayed_rd.reverse()
        return decayed_rd

    # 计算actor更新用的advantage value
    def advantage_calcu(self, decay_reward, state_t1, speed_):
        state_t1 = torch.Tensor(state_t1)
        speed_ = torch.Tensor(speed_)
        common_feature = self.common(state_t1, speed_)
        critic_value_ = self.v(common_feature)
        d_reward = torch.Tensor(decay_reward)
        advantage = d_reward - critic_value_
        return advantage

    # 计算critic更新用的 Q(s, a)和 V(s)
    def critic_update(self, state_t1, speed_, d_reward_):
        q_value = torch.Tensor(d_reward_).squeeze(-1)

        common_feature = self.common(state_t1, speed_)
        target_value = self.v(common_feature).squeeze(-1)
        critic_loss = self.c_loss(target_value, q_value)
        self.history_critic = critic_loss.detach().item()
        self.c_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.c_opt.step()

    def actor_update(self, state, speed_, action_acc, action_ori, advantage):
        action_acc = torch.FloatTensor(action_acc)
        action_ori = torch.FloatTensor(action_ori)
        self.a_opt.zero_grad()
        feature_common = self.common(state, speed_)
        feature_common_old = self.commonold(state, speed_)
        (pi_acc_m, pi_acc_s), (pi_ori_m, pi_ori_s) = self.pi(feature_common)
        (pi_acc_m_old, pi_acc_s_old), (pi_ori_m_old, pi_ori_s_old) = self.piold(feature_common_old)

        pi_dist_acc = Normal(pi_, pi_sigma)
        pi_dist_old = Normal(pi_mean_old, pi_sigma_old)

        logprob_ = pi_dist.log_prob(action_acc.reshape(-1, 1))
        logprob_old = pi_dist_old.log_prob(action_acc.reshape(-1, 1))

        ratio = torch.exp(logprob_ - logprob_old)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

        actor_loss = torch.min(torch.cat((surrogate1, surrogate2), dim=1), dim=1)[0]
        actor_loss = -torch.mean(actor_loss)
        self.history_actor = actor_loss.detach().item()

        actor_loss.backward(retain_graph=True)
        self.a_opt.step()

    def update(self, state, action_, discount_reward_):
        self.hard_update(self.pi, self.piold)
        state_ = torch.Tensor(state)
        act = action_
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_)

        for i in range(self.update_actor_epoch):
            self.actor_update(state_, act, adv)
            print(f'epochs: {self.ep}, timestep: {self.t}, actor_loss: {self.history_actor}')

        for i in range(self.update_critic_epoch):
            self.critic_update(state_, d_reward)
            print(f'epochs: {self.ep}, timestep: {self.t}, critic_loss: {self.history_critic}')

    @staticmethod
    def hard_update(model, target_model):
        weight_model = copy.deepcopy(model.state_dict())
        target_model.load_state_dict(weight_model)