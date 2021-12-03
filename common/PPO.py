import numpy
import torch
from net_builder import Common, Actor_builder, Critic_builder
from torch.distributions import Normal
from itertools import chain
from collections import deque
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import copy

LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
DECAY = 0.95
EPILSON = 0.2
torch.autograd.set_detect_anomaly(True)


class PPO:
    def __init__(self, in_channel, in_shape, action_space, batch_size):
        self.action_space = action_space
        self.input_channel = in_channel
        self.pixel_shape = in_shape
        self.batch_size = batch_size
        self._init(self.input_channel, self.batch_size)
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.decay_index = DECAY
        self.epilson = EPILSON
        self.c_loss = torch.nn.MSELoss()
        self.c_opt = torch.optim.Adam(params=chain(self.common.parameters(), self.v.parameters()), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=chain(self.common.parameters(), self.pi.parameters()), lr=self.lr_actor)
        self.update_actor_epoch = 3
        self.update_critic_epoch = 3
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
        self.commonold.eval()
        self.piold.eval()

    def get_action(self, obs_, speed_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        speed_ = torch.Tensor(copy.deepcopy(speed_))
        self.common.eval()
        self.pi.eval()
        common_feature = self.common(obs_, speed_)
        (acc_m, acc_s), (ori_m, ori_s) = self.pi(common_feature)
        # print(f'mu: {mean.cpu().item()}')

        # 增加1e-8防止正态分布计算时除法越界
        orie = Normal(ori_m.cpu().detach(), ori_s.cpu().detach() + 1e-8)
        accel = Normal(acc_m.cpu().detach(), acc_s.cpu().detach() + 1e-8)

        prob_ori = torch.clamp(orie.sample(), -0.5, 0.5)
        log_prob_ori = orie.log_prob(prob_ori)
        prob_accel = torch.clamp(accel.sample(), -0.3, 0.5)
        log_prob_accel = accel.log_prob(prob_accel)

        self.common.train()
        self.pi.train()

        return (prob_accel, log_prob_accel), (prob_ori, log_prob_ori)

    def state_store_memory(self, s, v, acc, ori, r, logprob_acc, logprob_ori):
        self.memory.append((s, v, acc, ori, r, logprob_acc, logprob_ori))

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
        torch.nn.utils.clip_grad_norm(chain(self.common.parameters(), self.v.parameters()), max_norm=5, norm_type=2)
        self.c_opt.step()

    def actor_update(self, state, speed_, action_acc, action_ori, advantage):
        with torch.autograd.detect_anomaly():
            action_acc = torch.FloatTensor(action_acc)
            action_ori = torch.FloatTensor(action_ori)

            feature_common = self.common(state, speed_)
            feature_common_old = self.commonold(state, speed_)
            (pi_acc_m, pi_acc_s), (pi_ori_m, pi_ori_s) = self.pi(feature_common)
            (pi_acc_m_old, pi_acc_s_old), (pi_ori_m_old, pi_ori_s_old) = self.piold(feature_common_old)

            if torch.any(torch.isnan(pi_ori_s)):
                print('invild value sigma pi')

            if torch.any(torch.isnan(pi_acc_s)):
                print('invild value sigma pi')
            # print(pi_acc_m, pi_acc_s)
            # print(pi_ori_m, pi_ori_s)

            # 增加1e-8防止正态分布计算时除法越界
            pi_dist_acc = Normal(pi_acc_m, pi_acc_s + 1e-8)
            pi_dist_acc_old = Normal(pi_acc_m_old, pi_acc_s_old + 1e-8)

            # 增加1e-8防止正态分布计算时除法越界
            pi_dist_ori = Normal(pi_ori_m, pi_ori_s + 1e-8)
            pi_dist_ori_old = Normal(pi_ori_m_old, pi_ori_s_old + 1e-8)

            logprob_acc = pi_dist_acc.log_prob(action_acc.reshape(-1, 1))
            logprob_acc_old = pi_dist_acc_old.log_prob(action_acc.reshape(-1, 1))

            logprob_ori = pi_dist_ori.log_prob(action_ori.reshape(-1, 1))
            logprob_ori_old = pi_dist_ori_old.log_prob(action_ori.reshape(-1, 1))

            ratio_acc = torch.exp(logprob_acc - logprob_acc_old)
            ratio_ori = torch.exp(logprob_ori - logprob_ori_old)

            if torch.any(torch.isnan(ratio_acc)) or torch.any(torch.isinf(ratio_acc)):
                print('invild value sigma pi')

            if torch.any(torch.isnan(ratio_ori)) or torch.any(torch.isinf(ratio_ori)):
                print('invild value sigma pi')

            print(ratio_acc)
            print(ratio_ori)

            # 切换ratio中inf值为固定值，防止inf进入backward计算
            ratio_ori = torch.where(torch.isinf(ratio_ori), torch.full_like(ratio_ori, 3), ratio_ori)

            surrogate1_acc = ratio_acc * advantage
            surrogate2_acc = torch.clamp(ratio_acc, 1-self.epilson, 1+self.epilson) * advantage

            surrogate1_ori = ratio_ori * advantage
            surrogate2_ori = torch.clamp(ratio_ori, 1-self.epilson, 1+self.epilson) * advantage

            acc_loss = torch.min(torch.cat((surrogate1_acc, surrogate2_acc), dim=1), dim=1)[0]
            ori_loss = torch.min(torch.cat((surrogate1_ori, surrogate2_ori), dim=1), dim=1)[0]

            self.a_opt.zero_grad()
            actor_loss = acc_loss + ori_loss
            actor_loss = -torch.mean(actor_loss)
            self.history_actor = actor_loss.detach().item()

            actor_loss.backward(retain_graph=True)
            self.a_opt.step()

    def update(self, state, speed_, action_acc, action_ori, discount_reward_):
        self.hard_update(self.pi, self.piold)
        self.hard_update(self.common, self.commonold)
        state_ = torch.Tensor(state)
        speed_cache = torch.Tensor(speed_)
        act_acc = action_acc
        act_ori = action_ori
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_, speed_cache).detach()

        for i in range(self.update_actor_epoch):
            self.actor_update(state_, speed_cache, act_acc, act_ori, adv)
            print(f'epochs: {self.ep}, timesteps: {self.t}, actor_loss: {self.history_actor}')

        for i in range(self.update_critic_epoch):
            self.critic_update(state_, speed_cache, d_reward)
            print(f'epochs: {self.ep}, timesteps: {self.t}, critic_loss: {self.history_critic}')

    def save_model(self, name):
        torch.save({'common': self.common.state_dict(),
                    'actor': self.pi.state_dict(),
                    'critic': self.v.state_dict(),
                    'opt_actor': self.a_opt.state_dict(),
                    'opt_critic': self.c_opt.state_dict()}, name)

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.common.load_state_dict(checkpoints['common'])
        self.pi.load_state_dict(checkpoints['actor'])
        self.v.load_state_dict(checkpoints['critic'])
        self.a_opt.load_state_dict(checkpoints['opt_actor'])
        self.c_opt.load_state_dict(checkpoints['opt_critic'])

    @staticmethod
    def hard_update(model, target_model):
        weight_model = copy.deepcopy(model.state_dict())
        target_model.load_state_dict(weight_model)

    @staticmethod
    def data_pcs(obs_: dict):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ',
                 'opponents',
                 'rpm',
                 'trackPos',
                 'wheelSpinVel',
                 'img',
                 'trackPos']
        # for i in range(len(names)):
        #     exec('%s = obs_[i]' %names[i])
        focus_ = obs_[0]
        speedX_ = obs_[1]
        speedY_ = obs_[2]
        speedZ_ = obs_[3]
        opponent_ = obs_[4]
        rpm_ = obs_[5]
        trackPos_ = obs_[6]
        wheelSpinel_ = obs_[7]
        img = obs_[8]
        trackPos = obs_[9]
        img_data = np.zeros(shape=(64, 64, 3))
        for i in range(3):
            # img_data[:, :, i] = 255 - img[:, i].reshape((64, 64))
            img_data[:, :, i] = img[:, i].reshape((64, 64))
        img_data = Image.fromarray(img_data.astype(np.uint8))
        img_data = np.array(img_data.transpose(Image.FLIP_TOP_BOTTOM))
        img_data = rgb2gray(img_data).reshape(1, 1, img_data.shape[0], img_data.shape[1])
        return focus_, speedX_, speedY_, speedZ_, opponent_, rpm_, trackPos_, wheelSpinel_, img_data, trackPos
