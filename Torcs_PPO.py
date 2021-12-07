#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
sys.path.append("./common/")
from skimage.color import rgb2gray
from common.gym_torcs import TorcsEnv
from common.PPO import PPO
import numpy as np
import platform
import os

torch.set_printoptions(sci_mode=False)


BATCH_SIZE = 16
MAX_STEP_EPISODE = 2000
TRAINABLE = True
VISION = True
DECAY = 0.95
VISION_SHAPE = (64, 64)
STATE_DIM = 8
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

    agent = PPO(state_dim=STATE_DIM, action_space=action_range, batch_size=BATCH_SIZE)
    epochs = 20000
    ep_history = []

    for ep in range(epochs):
        if np.mod(agent.ep, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            obs = env.reset(relaunch=True)
        else:
            obs = env.reset()
        ep_rh = 0

        # 数据维度初始化
        _, speedX, _, _, _, _, track, _, _, track_pos, angle = agent.data_pcs(obs)
        state_t = np.hstack((speedX, track_pos, angle, track[[0, 5, 9, 13, 18]]))

        for t in range(MAX_STEP_EPISODE):
            action_ori, logprob_ori_ = agent.get_action(state_t)

            # action = np.array((action_ori.detach().numpy(), action_acc.detach().numpy()), dtype='float')
            action = np.array((action_ori.detach().numpy()), dtype='float')
            obs_t1, reward, done, _ = env.step(action)

            if done and t < MAX_STEP_EPISODE - 1:
                reward = -3
            _, speedX_t1, _, _, _, _, track_t1, _, _, track_pos_t1, angle_t1 = agent.data_pcs(obs_t1)

            print(reward, logprob_ori_)

            state_t1 = np.hstack((speedX_t1, track_pos_t1, angle_t1, track[[0, 5, 9, 13, 18]]))

            ep_rh += reward

            agent.state_store_memory(state_t, action_ori.detach().numpy().reshape(-1, 1),
                                     reward, logprob_ori_)

            state_t = state_t1

            if (t+1) % agent.batch_size == 0 or t == (MAX_STEP_EPISODE - 1) or (done and (t+1) % agent.batch_size > 16):
                print(f't leangth: {t}')
                s_t, a_ori, rd, _ = zip(*agent.memory)
                s_t = np.stack(s_t, axis=0).squeeze()
                a_ori = np.concatenate(a_ori)
                rd = np.array(rd)

                discount_reward = agent.decayed_reward(state_t1, rd)

                agent.update(s_t, a_ori, discount_reward)
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
