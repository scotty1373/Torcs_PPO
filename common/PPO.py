import numpy
import torch
from net_simple import Actor_Model, Critic_Model
from torch.distributions import Normal
from collections import deque
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import copy

LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
DECAY = 0.99
EPILSON = 0.2
torch.autograd.set_detect_anomaly(True)


class PPO:
    def __init__(self, state_dim, action_space, batch_size):
        self.action_space = action_space
        self.state_dim = state_dim
        self.batch_size = batch_size
        self._init(self.state_dim, self.batch_size)
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.decay_index = DECAY
        self.epilson = EPILSON
        self.c_loss = torch.nn.MSELoss()
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor)
        self.update_actor_epoch = 3
        self.update_critic_epoch = 3
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, state_dim, train_batch):
        self.pi = Actor_Model(state_dim)
        self.piold = Actor_Model(state_dim)
        self.v = Critic_Model(state_dim)
        self.memory = deque(maxlen=train_batch)

    def get_action(self, obs_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        self.pi.eval()
        ac_mean, ac_sigma = self.pi(obs_)

        # 增加1e-8防止正态分布计算时除法越界
        orie = Normal(ac_mean.cpu().detach(), ac_sigma.cpu().detach() + 1e-8)

        prob_ori = torch.clamp(orie.sample(), -1, 1)
        log_prob_ori = orie.log_prob(prob_ori)
        self.pi.train()

        return prob_ori, ac_mean.cpu().detach().item()

    def state_store_memory(self, s, ori, r, logprob_ori):
        self.memory.append((s, ori, r, logprob_ori))

    # 计算reward衰减，根据马尔可夫过程，从最后一个reward向前推
    def decayed_reward(self, singal_state_frame, reward_):
        decayed_rd = []
        state_frame = torch.Tensor(singal_state_frame)
        value_target = self.v(state_frame).detach().numpy()
        for rd_ in reward_[::-1]:
            value_target = rd_ + value_target * self.decay_index
            decayed_rd.append(value_target)
        decayed_rd.reverse()
        return decayed_rd

    # 计算actor更新用的advantage value
    def advantage_calcu(self, decay_reward, state_t1):
        state_t1 = torch.Tensor(state_t1)
        critic_value_ = self.v(state_t1)
        d_reward = torch.Tensor(decay_reward)
        advantage = d_reward - critic_value_
        return advantage

    # 计算critic更新用的 Q(s, a)和 V(s)
    def critic_update(self, state_t1, d_reward_):
        q_value = torch.Tensor(d_reward_).squeeze(-1)

        target_value = self.v(state_t1).squeeze(-1)
        critic_loss = self.c_loss(target_value, q_value)
        self.history_critic = critic_loss.detach().item()
        self.c_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=1, norm_type=2)
        self.c_opt.step()

    def actor_update(self, state, action_ori, advantage):
        with torch.autograd.detect_anomaly():
            action_ori = torch.FloatTensor(action_ori)

            pi_ori_m, pi_ori_s = self.pi(state)
            pi_ori_m_old, pi_ori_s_old = self.piold(state)

            if torch.any(torch.isnan(pi_ori_s)):
                print('invalid value sigma pi')

            if torch.any(torch.isnan(pi_ori_m)):
                print('invalid value mean pi')

            # 增加1e-8防止正态分布计算时除法越界
            pi_dist_ori = Normal(pi_ori_m, pi_ori_s + 1e-8)
            pi_dist_ori_old = Normal(pi_ori_m_old.detach(), pi_ori_s_old.detach() + 1e-8)

            logprob_ori = pi_dist_ori.cdf(action_ori.reshape(-1, 1))
            logprob_ori_old = pi_dist_ori_old.cdf(action_ori.reshape(-1, 1))

            ratio_ori = logprob_ori / (logprob_ori_old + 1e-8)

            if torch.any(torch.isnan(ratio_ori)) or torch.any(torch.isinf(ratio_ori)):
                print('invalid value sigma pi')

            # 切换ratio中inf值为固定值，防止inf进入backward计算
            # ratio_ori = torch.where(torch.isinf(ratio_ori), torch.full_like(ratio_ori, 3), ratio_ori)

            surrogate1_ori = ratio_ori * advantage
            surrogate2_ori = torch.clamp(ratio_ori, 1-self.epilson, 1+self.epilson) * advantage
            actor_loss = torch.min(torch.cat((surrogate1_ori, surrogate2_ori), dim=1), dim=1)[0]

            self.a_opt.zero_grad()
            actor_loss = -torch.mean(actor_loss)

            actor_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=1, norm_type=2)

            # print(self.pi.ori_meanDense4.weight.grad)

            self.a_opt.step()
            self.history_actor = actor_loss.detach().item()

    def update(self, state, action_ori, discount_reward_):
        self.hard_update(self.pi, self.piold)
        state_ = torch.Tensor(state)
        act_ori = action_ori
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_).detach()

        for i in range(self.update_actor_epoch):
            self.actor_update(state_, act_ori, adv)
            print(f'epochs: {self.ep}, time_steps: {self.t}, actor_loss: {self.history_actor}')

        for i in range(self.update_critic_epoch):
            self.critic_update(state_, d_reward)
            print(f'epochs: {self.ep}, time_steps: {self.t}, critic_loss: {self.history_critic}')

    def save_model(self, name):
        torch.save({'actor': self.pi.state_dict(),
                    'critic': self.v.state_dict(),
                    'opt_actor': self.a_opt.state_dict(),
                    'opt_critic': self.c_opt.state_dict()}, name)

    def load_model(self, name):
        checkpoints = torch.load(name)
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
                 'trackPos',
                 'angle']
        # for i in range(len(names)):
        #     exec('%s = obs_[i]' %names[i])
        focus_ = obs_[0]
        speedX_ = obs_[1]
        speedY_ = obs_[2]
        speedZ_ = obs_[3]
        opponent_ = obs_[4]
        rpm_ = obs_[5]
        track = obs_[6]
        wheelSpinel_ = obs_[7]
        img = obs_[8]
        trackPos = obs_[9]
        angle = obs_[10]
        img_data = np.zeros(shape=(64, 64, 3))
        for i in range(3):
            # img_data[:, :, i] = 255 - img[:, i].reshape((64, 64))
            img_data[:, :, i] = img[:, i].reshape((64, 64))
        img_data = Image.fromarray(img_data.astype(np.uint8))
        img_data = np.array(img_data.transpose(Image.FLIP_TOP_BOTTOM))
        img_data = rgb2gray(img_data).reshape(1, 1, img_data.shape[0], img_data.shape[1])
        return focus_, speedX_, speedY_, speedZ_, opponent_, rpm_, track, wheelSpinel_, img_data, trackPos, angle
