# -*- coding: utf-8 -*-
import time
import torch
from torch import nn
from torch.distributions import Normal
from itertools import chain


class Common(nn.Module):
    def __init__(self, in_channel):
        super(Common, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32,
                               kernel_size=(5, 5), stride=(3, 3), padding=(0, 0))
        self.activation1 = nn.ReLU(inplace=True)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.activation2 = nn.ReLU(inplace=True)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.activation3 = nn.ReLU(inplace=True)
        self.Conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.activation4 = nn.ReLU(inplace=True)
        self.Dense1 = nn.Linear(6400, 512)
        self.Dense1act = nn.ReLU(inplace=True)
        self.position = nn.Linear(4, 128)

    def forward(self, x, y):
        common = self.Conv1(x)
        common = self.activation1(common)
        common = self.Conv2(common)
        common = self.activation2(common)
        common = self.Conv3(common)
        common = self.activation3(common)
        common = self.Conv4(common)
        common = self.activation4(common)
        common = torch.flatten(common, start_dim=1, end_dim=-1)

        down_shape_vector = self.Dense1(common)
        down_shape_vector = self.Dense1act(down_shape_vector)
        vehicle_pos = self.position(y)

        concat_vector = torch.cat([down_shape_vector, vehicle_pos], dim=1)
        return concat_vector


class Actor_builder(nn.Module):
    def __init__(self):
        super(Actor_builder, self).__init__()
        # self.common = nn.Linear(6528, 512)
        # self.common_act = nn.ReLU()

        self.acc_commonDense1 = nn.Linear(640, 256)
        self.acc_meanact1 = nn.ReLU()
        self.acc_meanDense2 = nn.Linear(256, 1)
        self.acc_meanout = nn.Tanh()
        self.acc_sigmaDense1 = nn.Linear(256, 1)
        self.acc_sigmaout = nn.Softplus()

        self.ori_commonDense1 = nn.Linear(640, 256)
        self.ori_meanact1 = nn.ReLU()
        self.ori_meanDense2 = nn.Linear(256, 1)
        torch.nn.init.uniform_(self.ori_meanDense2.weight, a=0, b=3e-3)
        self.ori_meanout = nn.Tanh()
        self.ori_sigmaDense1 = nn.Linear(256, 1)
        torch.nn.init.uniform_(self.ori_sigmaDense1.weight, a=-1e-3, b=1e-3)
        self.ori_sigmaout = nn.Softplus()

    def forward(self, common):
        # common_data = self.common(common)
        # common_data = self.common_act(common_data)

        acc_common = self.acc_commonDense1(common)
        acc_mean = self.acc_meanact1(acc_common)
        acc_mean = self.acc_meanDense2(acc_mean)
        acc_mean = self.acc_meanout(acc_mean)
        acc_sigma = self.acc_sigmaDense1(acc_common)
        acc_sigma = self.acc_sigmaout(acc_sigma)

        ori_common = self.ori_commonDense1(common)
        ori_mean = self.ori_meanact1(ori_common)
        ori_mean = self.ori_meanDense2(ori_mean)
        ori_mean = self.ori_meanout(ori_mean)
        ori_sigma = self.ori_sigmaDense1(ori_common)
        ori_sigma = self.ori_sigmaout(ori_sigma)
        return (acc_mean, acc_sigma), (ori_mean, ori_sigma)


class Critic_builder(nn.Module):
    def __init__(self):
        super(Critic_builder, self).__init__()
        self.vDense1 = nn.Linear(640, 256)
        self.vact1 = nn.ReLU(inplace=True)
        self.vDense2 = nn.Linear(256, 1)

    def forward(self, common):
        critic_common = self.vDense1(common)
        critic_value = self.vact1(critic_common)
        critic_value = self.vDense2(critic_value)
        return critic_value


class _built:
    def __init__(self, in_channel, out_dim):
        super(_built, self).__init__()
        self.common_ = Common(in_channel)
        self.actor_ = Actor_builder()
        self.critic_ = Critic_builder()

    def get_action(self, x, y):
        common_vector = self.common_(x, y)
        mean_, sigma_ = self.actor_(common_vector)
        dist = Normal(mean_, sigma_)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


if __name__ == "__main__":
    in_c = 4
    common_vt = Common(in_c)
    actor_vt = Actor_builder()
    critic_vt = Critic_builder()

    actor_opt = torch.optim.Adam(lr=1e-2, params=chain(common_vt.parameters(), actor_vt.parameters()))
    critic_opt = torch.optim.Adam(lr=1e-2, params=chain(common_vt.parameters(), critic_vt.parameters()))

    x = torch.randn((10, 4, 64, 64))
    y = torch.randn((10, 4))
    x_out = torch.randn((10, 1))

    for i in range(10):
        common_feature = common_vt(x, y)
        (acc_mean, acc_sigma), (ori_mean, ori_sigma) = actor_vt(common_feature)

        distribution_acc = Normal(acc_mean, acc_sigma)
        action_acc = distribution_acc.sample()
        log_prob_acc = distribution_acc.log_prob(action_acc)

        distribution_ori = Normal(ori_mean, ori_sigma)
        action_ori = distribution_ori.sample()
        log_prob_ori = distribution_ori.log_prob(action_ori)

        # loss_actor = torch.pow((log_prob_acc - x_out), 2) + torch.pow((log_prob_ori - x_out), 2)
        loss_actor = torch.pow((log_prob_acc - x_out), 2)

        '''
        由于actor网络使用的是两组相同的网络结构，但是两组网络参数不相同，acc用于输出加速减速动作，ori用于输出转角方向，
        acc和ori公用上层common层的特征提取结果，最后计算loss的时候一定满足链式法则，即正向计算acc的结果在反向传播过程中
        只会对acc网络权重进行更新，而不会对ori网络权重进行更新。
        所以在计算acc ori损失函数时可以将两个loss求和计算，最终BP过程只会对正向计算过程中的权重进行更新
        
        例如：
            给定loss为：
            loss_actor = torch.pow((log_prob_acc - x_out), 2)
            
            >>> actor_vt.ori_meanDense2.weight.grad
            PyDev console: starting.
            >>> actor_vt.acc_meanDense2.weight.grad
            tensor([[ 7.7915e-03,  1.4829e-02,  0.0000e+00,  4.0826e-01,  4.1580e-01,
                      2.8618e-01,  1.8830e-01, -2.1339e-03,  7.0063e-02,  2.2209e-01,
                      2.4431e-01,  5.9790e-02,  4.3749e-03,  5.7418e-02,  3.2877e-01,
                      3.5136e-02,  2.9982e-02, -6.2930e-03,  0.0000e+00,  8.3210e-02,
                     -1.8249e-03,  2.1643e-01,  1.8007e-02,  0.0000e+00,  2.2950e-01,
                      1.8979e-01,  1.2807e-01,  2.8301e-01,  3.5640e-01,  6.1736e-01,
                      0.0000e+00,  5.6895e-01, -1.0724e-02,  5.3026e-01,  2.9931e-02,
                      1.9439e-02,  6.9740e-02,  1.1615e-01,  3.9018e-01,  3.0434e-01,
                      2.3194e-01,  3.1673e-02,  2.9240e-02,  2.6205e-01,  8.9178e-02,
                      2.8489e-01,  2.8491e-01,  0.0000e+00,  1.2576e-01,  6.0632e-01,
                      6.4988e-02, -1.3984e-02,  1.7199e-01,  7.0150e-02,  2.8506e-01,
                      3.5482e-01,  0.0000e+00,  3.5265e-01,  7.9754e-02,  6.9129e-02,
                      7.7240e-02,  4.3274e-02,  2.9711e-02,  2.8011e-02,  1.9069e-01,
                      3.9594e-01,  9.4027e-02, -2.3064e-03,  2.8054e-02,  6.7224e-01,
                      2.4464e-01,  6.5268e-01,  3.3836e-01,  3.5965e-01,  2.8844e-02,
                      4.1631e-02,  6.9509e-02,  9.4453e-02,  2.3449e-02,  1.0800e-01,
                      4.1456e-02,  2.9170e-01,  5.4801e-03,  5.4831e-01,  3.2581e-01,
                      0.0000e+00, -4.9662e-03,  1.9376e-01,  2.1704e-01,  5.9088e-02,
                      1.7974e-01,  5.0418e-01,  3.7626e-01,  3.2790e-01,  7.2883e-01,
                      7.8809e-02,  1.9526e-01,  0.0000e+00,  3.0888e-01,  3.5117e-01,
                     -5.7925e-05,  4.1785e-01,  4.1091e-01,  1.7732e-01,  9.9811e-02,
                      0.0000e+00,  9.6404e-02,  3.1956e-01,  5.2171e-01,  1.5105e-01,
                      4.8497e-02,  0.0000e+00,  4.8336e-01,  2.3454e-01,  3.1647e-01,
                      4.3841e-02,  1.2299e-01,  7.6449e-01,  3.5621e-02,  0.0000e+00,
                      7.9285e-02,  0.0000e+00,  1.0953e-01,  0.0000e+00,  4.2473e-01,
                      8.1183e-02,  4.3464e-02,  1.4792e-01]])
            可见在loss只有acc情况下，ori权重没有grad的记录值
        '''

        loss_actor = torch.mean(loss_actor)
        actor_opt.zero_grad()
        loss_actor.backward(retain_graph=True)
        actor_opt.step()
        time.time()

    common_feature = common_vt(x, y)
    value = critic_vt(common_feature)
    loss_critic = torch.mean(torch.pow((value - x_out), 2))
    critic_opt.zero_grad()
    loss_critic.backward()
    critic_opt.step()
    time.time()
    # distribution = Normal(ori_mean, ori_sigma)
    # action_ori = distribution.sample()
    # log_prob_ori = distribution.log_prob(action_ori)
    #
    # loss_actor = torch.sqrt(log_prob_ori - x_out)
    # actor_opt.zero_grad()
    # loss_actor.backward()
    # actor_opt.step()

