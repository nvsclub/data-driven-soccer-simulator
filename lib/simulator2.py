import pandas as pd
from random import random
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
from tqdm import tqdm

import lib.draw as draw

# Defines
dev_neutralizer = 3
horizontal_scalling = 10.5/6.8

# Loading
def load_models():
    print('LOADING: [          ] 0%', end = '\r')
    passes = pickle.load(open('matrix/pass_gradient.sav', 'rb'))

    print('LOADING: [|         ] 10%', end = '\r')
    dribbles = pickle.load(open('matrix/dribble_gradient.sav', 'rb'))
    #dribble = None

    print('LOADING: [||        ] 20%', end = '\r')
    #shot_data = pd.read_csv('matrix/SFull.csv')
    #shot = NearestNeighbors(n_neighbors = 1)
    #shot.fit(shot_data[['x','y']], shot_data['xg'])
    shots = pickle.load(open('matrix/shot_gradient2.sav', 'rb'))

    print('LOADING: [||||||    ] 60%', end = '\r')
    #rebound_data = pd.read_csv('matrix/R.csv')
    #rebound = NearestNeighbors(n_neighbors = 1)
    #rebound.fit(rebound_data[['x','y']])
    #rebound_data = None
    #rebound = None
    print('Loaded simulation models')

    return passes, shots, dribbles#, rebound_data, rebound

passes, shots, dribbles = load_models()

# Simulator
## RBOUND NEEDS TO BE UPDATED
'''
class Rebound:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.success = False

    def simulate(self):
        instance = rebound_data.iloc[rebound.kneighbors([[self.x, self.y]])[1][0]]
        if random() < float(instance.xr):
            self.xr = float(np.random.normal(instance.xtarget - instance.dev_xx_xr * dev_neutralizer, instance.dev_xx_xr * dev_neutralizer))
            self.yr = float(np.random.normal(instance.ytarget, instance.dev_yy_xr * dev_neutralizer))
            if 0 < self.xr < 1 and 0 < self.yr < 1:
                self.success = True
                return min(self.x, self.xr), self.yr
        return self.x, self.y
'''

## DRIBBLE NEEDS TO BE UPDATED
class Dribble:
    def __init__(self, x, y, r, a, prevForwardAct, prevPass, prevDribble, action_reward, backwards_reward, out_of_bounds_reward):
        self.description = 'Dribble'
        self.color = 'C1'
        self.x, self.y = x, y
        self.r, self.a = r, a
        self.xt = self.x + self.r * np.cos((self.a - 0.5) * 2 * np.pi)
        self.yt = self.y + self.r * np.sin((self.a - 0.5) * 2 * np.pi)
        self.success = False
        self.rebound = None
        self.reward = 0
        
        # Settings
        self.action_reward = action_reward
        self.backwards_reward = backwards_reward
        self.out_of_bounds_reward = out_of_bounds_reward

        # Action memory
        self.prevForwardAct = prevForwardAct
        self.prevPass = prevPass
        self.prevDribble = prevDribble

    def simulate(self, verbose):
        self.reward = shots.predict_proba([[self.prevForwardAct, self.prevPass, self.prevDribble, self.x, self.y]])[0,1] / 10 + self.action_reward
        if (self.xt > 1) or (self.xt < 0) or (self.yt > 1) or (self.yt < 0):
            if verbose:
                print('Out of bounds dribble', self.success, 0, self.x, self.y, self.r, self.a)
            return self.xt, self.yt, self.out_of_bounds_reward

        prob = dribbles.predict_proba([[self.x, self.y, self.xt, self.yt]])[0,0]
        if random() < prob and (abs(self.x - self.xt) + abs(self.y - self.yt)) <= 0.35:
            self.success = True
            if verbose:
                print('Dribble', self.success, prob, self.x, self.y, self.xt, self.xt)
            if self.x > self.xt:
                return self.xt, self.yt, self.backwards_reward
            else:
                return self.xt, self.yt, self.reward
        else:
            if verbose:
                print('Dribble', self.success, prob, self.x, self.y, self.xt, self.xt)
            if self.x > self.xt:
                return self.xt, self.yt, self.backwards_reward
            else:
                return self.xt, self.yt, self.reward

# Action Pass:
## Requires x, y positional indicators and r, a intention indicators
## Call "simulate" after initializing to simulate the outcome
## "simulate" should only be called once for each instance
class Pass:
    def __init__(self, x, y, r, a, prevForwardAct, prevPass, prevDribble, action_reward, backwards_reward, out_of_bounds_reward):
        self.description = 'Pass'
        self.color = 'C0'
        self.x, self.y = x, y
        self.r, self.a = r, a
        self.xt = self.x + self.r * np.cos((self.a - 0.5) * 2 * np.pi)
        self.yt = self.y + self.r * np.sin((self.a - 0.5) * 2 * np.pi)
        self.success = False
        self.rebound = None
        self.reward = 0
        self.prob = 0

        # Settings
        self.action_reward = action_reward
        self.backwards_reward = backwards_reward
        self.out_of_bounds_reward = out_of_bounds_reward

        # Action memory
        self.prevForwardAct = prevForwardAct
        self.prevPass = prevPass
        self.prevDribble = prevDribble

    def simulate(self, verbose):
        self.reward = shots.predict_proba([[self.prevForwardAct, self.prevPass, self.prevDribble, self.x, self.y]])[0,1] / 10 + self.action_reward
        if (self.xt > 1) or (self.xt < 0) or (self.yt > 1) or (self.yt < 0):
            if verbose:
                print('Out of bounds pass', self.success, 0, self.x, self.y, self.r, self.a)
            return self.xt, self.yt, self.out_of_bounds_reward
        self.prob = passes.predict_proba([[self.x, self.y, self.r, self.a]])[0,1]
        if random() < self.prob:
            self.success = True
            if verbose:
                print('Pass', self.success, self.prob, self.x, self.y, self.r, self.a)
            if self.x > self.xt:
                return self.xt, self.yt, self.backwards_reward
            else:
                return self.xt, self.yt, self.reward
        else:
            if verbose:
                print('Pass', self.success, self.prob, self.x, self.y, self.r, self.a)
            if self.x > self.xt:
                return self.xt, self.yt, self.backwards_reward
            else:
                return self.xt, self.yt, self.reward

# Action Shot:
## Use global flag END_ON_XG to define binary outcome (False) or probabilistic outcome (True)
## Requires x, y positional indicators
## Call "simulate" after initializing to simulate the outcome
## "simulate" should only be called once for each instance
class Shot:
    def __init__(self, x, y, prevForwardAct, prevPass, prevDribble):
        self.description = 'Shot'
        self.color = 'C2'
        self.x, self.y = x, y
        self.r, self.a = 0, 0
        self.xt, self.yt = 1, 0.5
        self.success = False
        self.rebound = None
        self.xg = 0

        self.prevForwardAct = prevForwardAct
        self.prevPass = prevPass
        self.prevDribble = prevDribble

    def simulate(self, verbose):
        if self.x < 0.5:
            return self.x, self.y, 0
        self.xg = shots.predict_proba([[self.prevForwardAct, self.prevPass, self.prevDribble, self.x, self.y]])[0,1]
        
        if verbose: print('Shot', self.success, self.xg, self.x, self.y)

        if random() < self.xg:
            self.success = True
            return self.x, self.y, self.xg
        else:
            return self.x, self.y, self.xg
        
def evaluate_rebound_end(action):
    if action.rebound != None:
        if action.rebound.success:
            return False
        else:
            return True
    else:
        if action.success:
            return False
        else:
            return True

def evaluate_goal(action):
    '''if ENABLE_REBOUND:
        return action.success, action.success or not action.rebound.success'''
    return action.success, True



class Agent:
    def __init__(self, x, y, end_on_xg = True, enable_rebound = False, action_reward = 0.002, backwards_reward = -0.001, out_of_bounds_reward = -0.1, action_limit = None):
        self.x, self.y = x, y
        self.actions = []
        self.xg = 0
        self.goal = False
        self.end = False
        self.rewards = 0

        # Settings
        self.end_on_xg = end_on_xg
        self.enable_rebound = enable_rebound
        self.action_reward = action_reward
        self.backwards_reward = backwards_reward
        self.out_of_bounds_reward = out_of_bounds_reward
        self.action_limit = action_limit

        # Action memory
        self.prevForwardAct = False
        self.prevPass = False
        self.prevDribble = False
    
    def reset(self, x, y):
        self.x, self.y = x, y
        self.actions = []
        self.xg = 0
        self.goal = False
        self.end = False
        self.rewards = 0

    def do_shot(self, r=0, a=0, verbose=False):
        self.actions.append(Shot(self.x, self.y, self.prevForwardAct, self.prevPass, self.prevDribble))
        self.x, self.y, self.xg = self.actions[-1].simulate(verbose)
        self.rewards += self.xg
        self.goal, self.end = evaluate_goal(self.actions[-1])

        if self.end_on_xg:
            return [self.x, self.y], self.actions[-1].xg, self.end, [1, 0, r, a]

        return [self.x, self.y], self.xg, self.end, [1, 0, 0, r, a]
    
    def do_pass(self, r, a, verbose=False):
        self.actions.append(Pass(self.x, self.y, r, a, self.prevForwardAct, self.prevPass, self.prevDribble, action_reward = self.action_reward, backwards_reward = self.backwards_reward, out_of_bounds_reward = self.out_of_bounds_reward))
        self.x, self.y, reward = self.actions[-1].simulate(verbose)
        self.rewards += reward
        self.end = evaluate_rebound_end(self.actions[-1])

        if self.x > (self.x - np.cos(a)):
            self.prevForwardAct = True
        else:
            self.prevForwardAct = False
        self.prevPass = True
        self.prevDribble = False

        return [self.x, self.y], reward, self.end, [0, 1, 0, r, a]

    def do_dribble(self, r, a, verbose=False):
        self.actions.append(Dribble(self.x, self.y, r, a, self.prevForwardAct, self.prevPass, self.prevDribble, action_reward = self.action_reward, backwards_reward = self.backwards_reward, out_of_bounds_reward = self.out_of_bounds_reward))
        self.x, self.y, reward = self.actions[-1].simulate(verbose)
        self.rewards += reward
        self.end = evaluate_rebound_end(self.actions[-1])

        if self.x > (self.x - np.cos(a)):
            self.prevForwardAct = True
        else:
            self.prevForwardAct = False
        self.prevDribble = True
        self.prevPass = False

        return [self.x, self.y], self.goal, self.end, [0, 0, 1, r, a]

    def do_action(self, action, r=None, a=None, xt=None, yt=None, verbose=False):
        if not self.end:
            if xt is not None:
                r = np.sqrt((xt - self.x) ** 2 + (yt - self.y) ** 2)
                a = np.arctan2(yt - self.y, xt - self.x) / (2 * np.pi) + 0.5
            if action == 0 or action == 'Shot':
                next_obs, reward, is_done, action_used = self.do_shot(r, a, verbose)
                return next_obs, reward, is_done, action_used
            elif action == 1 or action == 'Pass':
                next_obs, reward, is_done, action_used = self.do_pass(r, a, verbose)
                return next_obs, reward, is_done, action_used
            elif action == 2 or action == 'Dribble':
                next_obs, reward, is_done, action_used = self.do_dribble(r, a, verbose)
                return next_obs, reward, is_done, action_used
        else:
            return None, self.out_of_bounds_reward, True, None
        

# Visualization
def draw_actions(agent, action_no):
    action_sample = agent.actions[action_no]
    x, xt, y, yt = action_sample.x * 100, action_sample.xt * 100, action_sample.y * 100, action_sample.yt * 100
    if xt - x == 0:
        if yt - y > 0:
            angle = 90
        else:
            angle = 270
    else:
        if xt - x > 0:
            angle = np.arctan((yt - y) / (xt - x)) * 360 / np.pi / 2
        else:
            angle = 180 + np.arctan((yt - y) / (xt - x)) * 360 / np.pi / 2
    distance = ((yt - y) ** 2 + ((xt - x)) ** 2) ** (1/2)

    plt.scatter(x, y, color = action_sample.color, zorder = 8 + 4 * action_no)

    all_patches = []
    all_patches.append(Wedge((x,y), distance, angle-0.5, angle+0.5, fc='#091442', zorder = 11 + 4 * action_no))
    all_patches.append(Arc((x,y), 3, 3*horizontal_scalling, 0, angle-45, angle+45, lw = 7 * action_sample.success, ec='#091442', zorder = 10 + 4 * action_no))
    all_patches.append(Arc((x,y), 3, 3*horizontal_scalling, 0, angle-45, angle+45, lw = 7, ec='#a2b3ff', zorder = 9 + 4 * action_no))

    for patch in all_patches:
        plt.gcf().gca().add_artist(patch)

def draw_actions_for_set(agent_set, action_no):
    action_set = []
    success_count = 0
    for agent in agent_set:
        if action_no < len(agent.actions):
            action_set.append(agent.actions[action_no])


            if agent.actions[action_no].description == 'Shot':
                success_count += agent.actions[action_no].xg
            elif agent.actions[action_no].success:
                success_count += 1

    action_sample = action_set[0]
    x, xt, y, yt = action_sample.x * 100, action_sample.xt * 100, action_sample.y * 100, action_sample.yt * 100
    if xt - x == 0:
        if yt - y > 0:
            angle = 90
        else:
            angle = 270
    else:
        if xt - x > 0:
            angle = np.arctan((yt - y) / (xt - x)) * 360 / np.pi / 2
        else:
            angle = 180 + np.arctan((yt - y) / (xt - x)) * 360 / np.pi / 2
    distance = ((yt - y) ** 2 + ((xt - x)) ** 2) ** (1/2)

    plt.scatter(x, y, color = action_sample.color, zorder = 8 + 4 * action_no)

    all_patches = []
    all_patches.append(Wedge((x,y), distance, angle-0.5, angle+0.5, fc='#091442', zorder = 11 + 4 * action_no))
    all_patches.append(Arc((x,y), 3, 3*horizontal_scalling, 0, angle-45, angle+45, lw = 7 * success_count/len(action_set), ec='#091442', zorder = 10 + 4 * action_no))
    all_patches.append(Arc((x,y), 3, 3*horizontal_scalling, 0, angle-45, angle+45, lw = 7, ec='#a2b3ff', zorder = 9 + 4 * action_no))

    for patch in all_patches:
        plt.gcf().gca().add_artist(patch)

def draw_play(agent, dpi = 144, directory='__default', plot_now=False, current_action_color = 'C0'):
    draw.pitch(dpi = dpi)
    for action in range(len(agent.actions)):
        draw_actions(agent, action)
    if not agent.end:
        plt.scatter(agent.x * 100, agent.y * 100, color = current_action_color, zorder = 100)
    if plot_now:
        return
    plt.savefig('img/' + directory + '.png')
    plt.clf()

def draw_play_from_agent_set(agent_set, dpi=144, plot_now=True, directory='a.png'):
    draw.pitch(dpi=dpi)
    max_len = 0
    for agent in agent_set:
        if len(agent.actions) > max_len:
            max_len = len(agent.actions)
    for action in range(max_len):
        draw_actions_for_set(agent_set, action)
    if plot_now:
        return
    plt.savefig(directory, format='png')
    plt.clf()