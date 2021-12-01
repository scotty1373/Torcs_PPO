#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys
sys.path.append("./common/")
from collections import deque
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from common.gym_torcs import TorcsEnv
import torch
import numpy as np
import platform
import time
import os

LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.0001
BATCH_SIZE = 32
MAX_STEP_EPISODE = 1000
TRAINABLE = True
VISION = True
DECAY = 0.99
CHANNEL = 1


if platform.system() == 'windows':
    temp = os.getcwd()
    CURRENT_PATH = temp.replace('\\', '/')
else:
    CURRENT_PATH = os.getcwd()
CURRENT_PATH = os.path.join(CURRENT_PATH, 'save_Model')
if not os.path.exists(CURRENT_PATH):
    os.makedirs(CURRENT_PATH)


class DDPG_NET:
    def __init__(self, shape_in, num_output, accele_range, angle_range):
        self.input_shape = shape_in
        self.out_shape = num_output
        self.learning_rate_a = LEARNING_RATE_ACTOR
        self.learning_rate_c = LEARNING_RATE_CRITIC
        self.memory = deque(maxlen=BATCH_SIZE)
        self.channel = CHANNEL
        self.train_start = 300
        self.batch_size = 128
        self.gamma = 0.9
        self.sigma_fixed = 3
        self.critic_input_action_shape = 1
        self.angle_range = angle_range
        self.accele_range = accele_range
        self.actor_model = self.actor_net_builder()
        self.critic_model = self.critic_net_build()
        self.actor_target_model = self.actor_net_builder()
        self.critic_target_model = self.critic_net_build()

        # self.actor_target_model.trainable = False
        # self.critic_target_model.trainable = False

        self.actor_history = []
        self.critic_history = []
        self.reward_history = []
        self.weight_hard_update()

    def state_store_memory(self, s, focus_, speedx, a, r, s_t1, focus_t1_, speedx_t1):
        self.memory.append((s, focus_, speedx, a, r, s_t1, focus_t1_, speedx_t1))


    @staticmethod
    def image_process(obs):
        origin_obs = rgb2gray(obs)
        car_shape = origin_obs[44:84, 28:68].reshape(1, 40, 40, 1)
        state_bar = origin_obs[84:, 12:]
        right_position = state_bar[6, 36:46].reshape(-1, 10)
        left_position = state_bar[6, 26:36].reshape(-1, 10)
        car_range = origin_obs[44:84, 28:68][22:27, 17:22]

        return car_shape, right_position, left_position, car_range

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
            img_data[:, :, i] = 255 - img[:, i].reshape((64, 64))
        img_data = rgb2gray(img_data/255).reshape(1, img_data.shape[0], img_data.shape[1], 1)
        return focus_, speedX_, speedY_, speedZ_, opponent_, rpm_, trackPos_, wheelSpinel_, img_data, trackPos

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_data = random.sample(self.memory, self.batch_size)
        s_, focus_, speedX_, a_, r_, s_t1_, focus_t1_, speedX_t1_ = zip(*batch_data)
        s_ = np.array(s_, dtype='float').squeeze(axis=1)
        focus_ = np.array(focus_, dtype='float').squeeze()
        speedX_ = np.array(speedX_t1_, dtype='float').squeeze()
        a_ = np.array(a_, dtype='float').squeeze(axis=2)   # ang = a[:, 0, :], acc = a[:, 1, :]

        r_ = np.array(r_, dtype='float').reshape(self.batch_size, -1)

        s_t1_ = np.array(s_t1_, dtype='float').squeeze(axis=1)
        focus_t1_ = np.array(focus_t1_, dtype='float').squeeze()
        speedX_t1_ = np.array(speedX_t1_, dtype='float').squeeze()
        # parameters initiation

        optimizer_actor = keras.optimizers.Adam(-self.learning_rate_a)
        optimizer_critic = keras.optimizers.Adam(self.learning_rate_c)

        with tf.GradientTape() as tape:
            q_target, q_real = self.critic_loss(s_, speedX_, r_, s_t1_, speedX_t1_, a_)
            # td-error
            loss = tf.reduce_mean(tf.square(q_target - q_real))
        grad_critic_loss = tape.gradient(loss, agent.critic_model.trainable_weights)
        optimizer_critic.apply_gradients(zip(grad_critic_loss, agent.critic_model.trainable_weights))

        with tf.GradientTape(persistent=True) as tape_actor:
            a = self.actor_model([s_, speedX_])
            a_ang, a_acc = tf.split(a, 2, axis=0)
            q = self.critic_model([s_, speedX_, tf.squeeze(a_ang, axis=[0]), tf.squeeze(a_acc, axis=[0])])
            actor_loss = tf.reduce_mean(q)
        policy_gradient = tape_actor.gradient(q, a)
        # q based on a, so gradient with the weight of actor is based on the chain rule
        grad_a = tape_actor.gradient(actor_loss, agent.actor_model.trainable_weights)
        optimizer_actor.apply_gradients(zip(grad_a, agent.actor_model.trainable_weights))

        agent.weight_soft_update()    # soft update should be not too lang and not too short
        agent.sigma_fixed *= .995     # decay normal function sigma val


if __name__ == '__main__':

    env = TorcsEnv(vision=VISION, throttle=True)

    test_train_flag = TRAINABLE

    action_shape = env.action_space.shape
    state_shape = np.array(env.observation_space.shape)
    action_range = env.action_space.high            # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    agent = DDPG_NET((64, 64, 4), np.ndim(action_shape), action_range[0], action_range[1])
    epochs = 400
    timestep = 0
    count = 0

    while True:
        if np.mod(count, 25) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        ep_history = np.array([])
        acc_ang_flag = 0
        live_time = 0

        focus, speedX, _, _, _, _, track, _, obs, _ = agent.data_pcs(ob)
        obs = np.stack((obs, obs, obs, obs), axis=3).reshape((1, 64, 64, -1))
        speedX = np.array(speedX).reshape(1)
        speedX = np.stack((speedX, speedX, speedX, speedX), axis=1)

        for index in range(MAX_STEP_EPISODE):
            ang_net, acc_net = agent.action_choose([obs, speedX])
            if count < 3000:
                ang_noise = agent.OU_angle.noise()
                acc_noise = agent.OU_accele.noise()
                ang_net = tf.add(ang_net, ang_noise)
                acc_net = tf.add(acc_net, acc_noise)
                agent.OU_angle.sigma *= 0.995
                agent.OU_accele.sigma *= 0.995
                ang_net = np.clip(ang_net, -action_range[0] + 0.5, action_range[0] - 0.5)
                acc_net = np.clip(acc_net, 0, action_range[1] - 0.5)
            else:
                pass

            action = np.array((ang_net, acc_net - 0.3), dtype='float')
            ob_t1, reward, done, _ = env.step(action)
            focus_t1, speedX_t1, _, _, _, _, track_t1, _, obs_t1, _ = agent.data_pcs(ob_t1)
            obs_t1 = np.append(obs[:, :, :, 1:], obs_t1, axis=3)
            speedX_t1 = np.reshape(speedX_t1, (1, 1))
            speedX_t1 = np.append(speedX[:, 1:], speedX_t1, axis=1)

            c_v = agent.critic_model([obs_t1, speedX_t1, ang_net, acc_net])
            c_v_target = agent.critic_target_model([obs_t1, speedX_t1, ang_net, acc_net])

            if done:
                agent.state_store_memory(obs, focus, speedX, [ang_net, acc_net], reward, obs_t1, focus_t1, speedX_t1)
                print(f'terminated by environment, timestep: {timestep},'
                      f'epoch: {count}, reward: {reward}, angle: {tf.squeeze(ang_net)},'
                      f'acc: {tf.squeeze(acc_net)}, reward_mean: {np.array(ep_history).sum()}')
                break

            ep_history = np.append(ep_history, reward)
            agent.state_store_memory(obs, focus, speedX, [ang_net, acc_net], reward, obs_t1, focus_t1, speedX_t1)

            if test_train_flag is True:
                agent.train_replay()

            print(f'timestep: {timestep},'
                  f'epoch: {count}, reward: {reward}, ang: {ang_net} '
                  f'acc: {acc_net}, reward_mean: {np.array(ep_history).sum()} '
                  f'c_r: {c_v}, c_t: {c_v_target}, live_time: {live_time} '
                  f'sigma: {agent.sigma_fixed}')

            timestep += 1
            obs = obs_t1
            speedX = speedX_t1
            live_time += 1
            time.sleep(0.1)

        if count == epochs:
            break
        elif count % 100 == 0:
            timestamp = time.time()
            agent.actor_model.save(CURRENT_PATH + '/' + f'action_model{timestamp}.h5')
            agent.critic_model.save(CURRENT_PATH + '/' + f'critic_model{timestamp}.h5')
        count += 1

    env.end()
    sys.exit()
