#  Original by https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import os
import ptan
import time
import argparse
import numpy as np
from random import random, sample
from collections import namedtuple

import lib.draw as draw
import lib.simulator as simulator
import lib.model as model
import lib.common as common

import torch
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define attempt and create directory to save progress
ATTEMPT = "ddpg1"
ENABLE_VIZ = True
save_path = os.path.join("tmp", "ddpg-" + ATTEMPT)
os.makedirs(save_path, exist_ok=True)
try:
    os.mkdir('img/decision_maps_' + ATTEMPT)
except:
    pass

# Define action spaces
OBSERVATION_SPACE = 2
ACTION_SPACE = 4

# Noise for actions
MIN_PROB_EXPLORATION = 0.0
ACTION_NOISE_MEAN = 0.0
ACTION_NOISE_STD = 0.0

# Algo parameters
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 0.001
REPLAY_SIZE = 100000
REPLAY_INITIAL = BATCH_SIZE
TEST_ITERS = 100
device = torch.device('cpu')

# Function to test the agent
def test_net(net, count = 100, device="cpu"):
    test_agent = simulator.Agent(random(), random())

    rewards = 0.0
    steps = 0

    # Run the agent COUNT times
    for _ in range(count):
        test_agent.reset(random(), random())
        # Run agent till it stops
        while True:
            # Calculate and do action
            obs = [test_agent.x, test_agent.y]
            
            action = net(torch.tensor(obs, dtype=torch.float).to(device)).cpu().detach().numpy()
            discrete_action = softmax_function(action[:2])
            r, a = action[-2], action[-1]
            
            next_obs, reward, is_done, action_used = test_agent.do_action(discrete_action, r, a)

            rewards += reward
            steps += 1
            # If done proceed to next try
            if is_done:
                break
        
    return rewards / count, steps / count

# Create buffer auxiliars
Experience = namedtuple('Episode', field_names=['state', 'action', 'reward', 'last_state', 'done'])

# Initialize simulator
sim = simulator.Agent(random(), random())

# Initialize networks and inteligent agents
act_net = model.DDPGActor(OBSERVATION_SPACE, ACTION_SPACE).to(device)
crt_net = model.DDPGCritic(OBSERVATION_SPACE, ACTION_SPACE).to(device)
tgt_act_net = ptan.agent.TargetNet(act_net)
tgt_crt_net = ptan.agent.TargetNet(crt_net)

agent = model.AgentDDPG(act_net, device=device)

act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

# Define soft_max function for discrete actions
def softmax_function(values):
    return_values = [max(MIN_PROB_EXPLORATION, np.exp(value)/np.exp(values).sum()) for value in values]
    return np.random.choice(len(return_values), p=return_values/sum(return_values))

buffer = []
iteration = 0
best_reward = None
iteration_stats = []
# Run main
while True:
    # Increase counter
    iteration += 1

    # Calculate and do action
    obs = [sim.x, sim.y]
    action = act_net(torch.tensor(obs, dtype=torch.float).to(device)).cpu().detach().numpy()
    discrete_action = softmax_function(action[:2])
    r, a = action[-2] + np.random.normal(ACTION_NOISE_MEAN, ACTION_NOISE_STD), action[-1] + np.random.normal(ACTION_NOISE_MEAN, ACTION_NOISE_STD)
    
    next_obs, reward, is_done, action_used = sim.do_action(discrete_action, r, a)
    
    # If agent ended, reset for next iteration
    if is_done:
        sim.reset(random(), random())
        if action_used is None:
            continue

    # Add experience to the buffer
    buffer.append(Experience(obs, action_used, reward, next_obs, is_done))
    if len(buffer) < REPLAY_INITIAL:
        continue

    # Sample and unpack actions
    batch = sample(buffer, BATCH_SIZE)
    states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

    # Train critic
    crt_opt.zero_grad()
    q_v = crt_net(states_v, actions_v)
    last_act_v = tgt_act_net.target_model(last_states_v)
    q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
    q_last_v[dones_mask] = 0.0
    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
    critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
    critic_loss_v.backward()
    crt_opt.step()

    # Train actor
    act_opt.zero_grad()
    cur_actions_v = act_net(states_v)
    actor_loss_v = -crt_net(states_v, cur_actions_v)
    actor_loss_v = actor_loss_v.mean()
    actor_loss_v.backward()
    act_opt.step()

    # Synchronize target networks
    tgt_act_net.alpha_sync(alpha=1 - 1e-3)
    tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

    # Register process every TEST_ITERS iterations
    if iteration % TEST_ITERS == 0:
        rewards, steps = test_net(act_net, device=device)
        print("%d, reward %.5f, steps %d" % (iteration, rewards, steps))
        iteration_stats.append([iteration, rewards, steps])
        pd.DataFrame(iteration_stats).to_csv('tmp/log' + ATTEMPT + '.csv')
        if (best_reward is None or best_reward < rewards) and ENABLE_VIZ:
            # Progress visualizer
            viz_df = pd.DataFrame([[i, j] for i in np.arange(0, 1.01, 0.01) for j in np.arange(0, 1, 0.015)], columns = ['x','y'])
            viz_df[['Shot', 'Pass', 'r', 'a']] = act_net(torch.FloatTensor(viz_df[['x','y']].to_numpy())).cpu().detach().numpy()
            viz_df['x'] *= 100
            viz_df['y'] *= 100

            ## Discrete action decisons
            draw.pitch()
            shot_action = viz_df[viz_df.Shot > viz_df.Pass]
            plt.scatter(shot_action.x, shot_action.y, s = 15, c = 'C0', alpha = 0.5, marker = 's', linewidth=0, zorder = 10)
            pass_action = viz_df[viz_df.Shot < viz_df.Pass]
            plt.scatter(pass_action.x, pass_action.y, s = 15, c = 'C1', alpha = 0.5, marker = 's', linewidth=0, zorder = 10)
            plt.savefig('img/decision_maps_' + ATTEMPT + '/passshotmap_' + str(iteration) + '.png', dpi = 900)
            plt.clf()

            ## Continuous action decisions
            draw.pitch()
            for i, row in viz_df.iterrows():
                if i % 3 == 0:
                    plt.arrow(row['x'], row['y'], row.r/abs(row.r) * np.cos((row['a'] - 0.5) * 2 * np.pi), row.r/abs(row.r) * np.sin((row['a'] - 0.5) * 2 * np.pi), length_includes_head = True, head_width = .5, head_length = .5, color = 'white')
            viz_df['i'] = [i for i in range(len(viz_df))]
            plt.scatter(viz_df[(viz_df.i % 3) == 0].x, viz_df[(viz_df.i % 3) == 0].y, s=(viz_df[(viz_df.i % 3) == 0].r + 1) ** 2, zorder = 99)
            plt.savefig('img/decision_maps_' + ATTEMPT + '/orientationmap_' + str(iteration) + '.png', dpi = 900)
            plt.clf()

            # Store best performers
            if best_reward is not None:
                print("Best reward updated: %.5f -> %.5f" % (best_reward, rewards))
                name = "best_%+.5f_%d.dat" % (rewards, iteration)
                fname = os.path.join(save_path, name)
                torch.save(act_net.state_dict(), fname)
            best_reward = rewards

