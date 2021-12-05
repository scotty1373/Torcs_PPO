#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys

import torch

sys.path.append("./common/")
from collections import deque
from skimage.color import rgb2gray
from common.gym_torcs import TorcsEnv
from common.PPO import PPO
import numpy as np
import platform
import time
import os

torch.set_printoptions(sci_mode=False)


BATCH_SIZE = 16
MAX_STEP_EPISODE = 1000
TRAINABLE = True
VISION = True
DECAY = 0.95
VISION_SHAPE = (64, 64)
CHANNEL = 4


if platform.system() == 'windows':
    temp = os.getcwd()
    CURRENT_PATH = temp.replace('\\', '/')
else:
    CURRENT_PATH = os.getcwd()
CURRENT_PATH = os.path.join(CURRENT_PATH, 'save_Model')
if not os.path.exists(CURRENT_PATH):
    os.makedirs(CURRENT_PATH)


def image_process(obs_data):
    origin_obs = rgb2gray(obs_data)
    car_shape = origin_obs[44:84, 28:68].reshape(1, 40, 40, 1)
    state_bar = origin_obs[84:, 12:]
    right_position = state_bar[6, 36:46].reshape(-1, 10)
    left_position = state_bar[6, 26:36].reshape(-1, 10)
    car_range = origin_obs[44:84, 28:68][22:27, 17:22]

    return car_shape, right_position, left_position, car_range


if __name__ == '__main__':
    env = TorcsEnv(vision=VISION, throttle=False)

    test_train_flag = TRAINABLE

    action_shape = env.action_space.shape
    state_shape = np.array(env.observation_space.shape)
    action_range = env.action_space.high            # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    agent = PPO(in_channel=CHANNEL, in_shape=VISION_SHAPE, action_space=action_range, batch_size=BATCH_SIZE)
    epochs = 2000
    ep_history = []

    for ep in range(epochs):
        if np.mod(agent.ep, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            obs = env.reset(relaunch=True)
        else:
            obs = env.reset()
        ep_rh = 0

        # 数据维度初始化
        _, _, _, _, _, _, track, _, image, speedX = agent.data_pcs(obs)
        state_t = np.stack((image, image, image, image), axis=0).reshape((1, -1, 64, 64))
        speedX = np.array(speedX).reshape(1)
        speedX = np.stack((speedX, speedX, speedX, speedX), axis=1)

        for t in range(MAX_STEP_EPISODE):
            (action_acc, logprob_acc_), (action_ori, logprob_ori_) = agent.get_action(state_t, speedX)

            # action = np.array((action_ori.detach().numpy(), action_acc.detach().numpy()), dtype='float')
            action = np.array((action_ori.detach().numpy()), dtype='float')
            obs_t1, reward, done, _ = env.step(action)

            if done and t < MAX_STEP_EPISODE - 1:
                reward = -10
            focus_t1, _, _, _, _, _, track_t1, _, image_t1, speedX_t1 = agent.data_pcs(obs_t1)
            state_t1 = np.append(image_t1, state_t[:, :3, :, :], axis=1)
            speedX_t1 = np.reshape(speedX_t1, (1, 1))
            speedX_t1 = np.append(speedX_t1, speedX[:, :3], axis=1)
            # print(action_ori)
            ep_rh += reward

            agent.state_store_memory(state_t, speedX, action_acc.detach().numpy().reshape(-1, 1),
                                     action_ori.detach().numpy().reshape(-1, 1), reward,
                                     logprob_acc_, logprob_ori_)

            state_t = state_t1
            speedX = speedX_t1

            if (t+1) % agent.batch_size == 0 or t == (MAX_STEP_EPISODE - 1) or (done and (t+1) % agent.batch_size > 16):
                print(f't leangth: {t}')
                s_t, sx_t, a_acc, a_ori, rd, _, _ = zip(*agent.memory)
                s_t = np.concatenate(s_t).squeeze()
                sx_t = np.concatenate(sx_t).squeeze()
                a_acc = np.concatenate(a_acc).squeeze()
                a_ori = np.concatenate(a_ori).squeeze()

                discount_reward = agent.decayed_reward(state_t1, speedX_t1, rd)

                agent.update(s_t, sx_t, a_acc, a_ori, discount_reward)
                agent.memory.clear()

            agent.t += 1

            if done:
                print('env done!!!')
                agent.memory.clear()
                break
        ep_history.append(ep_rh)
        agent.ep += 1
        print(f'epoch: {ep}, timestep: {agent.t}, reward_summary: {ep_rh}, ')

    env.end()
    sys.exit()
