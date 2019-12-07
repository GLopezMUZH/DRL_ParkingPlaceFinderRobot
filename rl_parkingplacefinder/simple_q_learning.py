
# %%
"""
    parkingplacefinder is an OpenSource python package for the reinforcement learning
    of an agent that searches for the most optimal free parking spot in a parking lot

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
"""
#%%
import numpy as np
import collections
import matplotlib.pylab as plt
import networkx as nx
from PIL import Image
import cv2
import scipy.misc
import csv
from tqdm import tqdm

import os
os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import Parking_lot
from Parking_lot import Parking_Lot
from Park_Finder_Agent import Park_Finder_Agent
from Park_Finder_Agent import Reward_Parameters

import time
from datetime import datetime

#%%
"""
#TODO:
    - Balance rewards
    - What is the best option to combine optimization for 1) reward of parking lot 2) walking distance 3) parking distance
    - Enable rendering with agent on the terminal state for better understanding
    - After maximizing reward (very early) agent somehow still goes for smaller rewards and sticks to close parking slots in the end
    - Allow agent only to drive straight into parking lot
"""
#%%

def print_frames(frames):
    for i, frame in enumerate(frames):
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Resulting state: {frame['resulting state']}")
        print(f"Path: {frame['state history']}")
        print(f"Action history: {frame['action history']}")
        print(f"Reward history: {frame['reward history']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        print(f"Found parking: {frame['new start']}")
        print(f"Walking distance: {frame['walk distance']}")
        print(f"Driving distance: {frame['drive distance']}")
        print('-------------')



def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    # if all the values are 0 in the Q table, pick random an action (otherwise it would always chose the first one)
    if sum(values)==0:
        # print("picking random")
        action = np.random.randint(0,4)
    else:
        # if there is already something learned, pick the one which has the highest reward attached to it
        action = np.argmax(values)
    return actions[action]

def make_combo_plot(a,b):
    fig, ax1 = plt.subplots()
    color = 'tab:green'
    ax1.set_xlabel('episodes (s)')
    ax1.set_ylabel('rewards', color=color)
    ax1.plot(a, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('epsilon', color=color)
    ax2.plot(b, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()



#%%
if __name__ == '__main__':

    ffp = Parking_lot.Filling_Function_Parameters(uniform_distribution_p_value = 0.5)
    ldp = Parking_lot.Lane_Direction_Parameters()
    EPISODES = 100 #55000
    show = False #True

    parking_environment = Parking_Lot(lane_direction_paramenters=ldp,
                            filling_function_parameters=ffp,
                            nr_parking_slots_per_lane=5,
                            nr_parking_lanes=4,
                            parking_lane_depth=2,
                            debug=True,
                            draw_graph = True,
                            show_summary = False
                            )

    parking_lot = parking_environment.get_env()

    reward_parameters = Reward_Parameters()
 
    env = Park_Finder_Agent(reward_parameters=reward_parameters, parking_environment=parking_environment)
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0
    frames = [] # for information
    fig = plt.figure()

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numEpisodes = EPISODES
    totalRewards = np.zeros(numEpisodes)
    learningRewards = np.zeros(numEpisodes)
    epsilon = np.zeros(numEpisodes)
    last_rewards = collections.deque(maxlen=200)

    pbar = tqdm(range(numEpisodes))
    for i in pbar:
        observation = env.agentPosition
        # Every xth episode we reset the car to position 0 and start again
        if i % (EPISODES/100) == 0:
            # print('starting episode ', i)
            observation = env.reset()
        done = False

        epRewards = 0
        history = []
        action_history = []
        reward_history = []


        while not done:
            finish = False
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1-EPS) else env.actionSpaceSample()
            # this ovservatio is a trial step
            observation_, reward, done, info = env.step(action)

            # making sure that when parking lot was reached or a crash with a parked car occures, we terminate this episode (just experimental, should be handled in the getReward function)
            resulting_state = observation+env.actionSpace[action]
            # print(env.actionSpace[action])

            #if reward == -(nx.shortest_path_length(parking_lot,source=env.agentPosition,target=max(env.vacant_list)))**2/max(env.vacant_list) or reward == reward_parameters.PARK_CRASH_REWARD or reward == reward_parameters.WALL_CRASH_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            if done:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0x7F == ord('q'):
                    break

            if observation_+env.actionSpace[action] in env.vacant_list and action == 5:
                finish = True

            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + GAMMA*Q[observation_,action_] - Q[observation,action])
            history.append(observation)
            action_history.append(action)
            reward_history.append(epRewards)
            last_rewards.append(epRewards)
            observation = observation_

            if resulting_state in env.vacant_list and action == 5:
                # print('Found parking lot in episode: {}, parked on {} which is {} and got a reward of {}'.format(i, resulting_state,parking_lot.nodes[resulting_state]['occupation'],epRewards))
                pbar.set_description("Reward %s " % round(epRewards, 2))

                walk_distance = nx.shortest_path_length(parking_lot,source=resulting_state,target=max(env.parking_lot.nodes))
                drive_distance = nx.shortest_path_length(parking_lot,source=0,target=resulting_state)
            if finish:
                print("start a new episode")
                done = True
                env.reset()
                history.append(resulting_state)
                frames.append({'state': observation, 'resulting state': resulting_state, 'state history': history[:-1],
                               'action history': action_history, 'reward history': reward_history,
                               'action': action, 'reward': epRewards, 'new start': done, 'walk distance': walk_distance,
                               'drive distance': drive_distance})
            else:
                walk_distance = False
                drive_distance = False

            if show:
                img = scipy.misc.toimage(env.grid)
                img = img.resize((500, 500))  # resizing so we can see our agent in all its glory.
                cv2.imshow("Parking Agent", np.array(img))

        if EPS - 2 / numEpisodes > 0:
            EPS -= 2 / numEpisodes
        else:
            EPS = 0
        # print("Epsilon: ", round(EPS, 3))
        # pbar.set_description("Epsilon %s" % round(EPS,2))
        epsilon[i] = EPS
        if epRewards:
            learningRewards[i] = epRewards
            reward_history.append(epRewards)

        if len(last_rewards) == 200 and len(list(set(last_rewards))) == 1:
            print("Early stopping because of no changes")
            epsilon = epsilon[epsilon != 0]
            learningRewards = learningRewards[learningRewards != 0]
            break


        # totalRewards[i] = epRewards

    print(print_frames(frames))
    # plt.plot(totalRewards)
    make_combo_plot(learningRewards,epsilon)

    # plt.plot(learningRewards)
    # plt.show()
    
    # print Q table
    first2pairs = {k: Q[k] for k in sorted(Q.keys())[:10]}
    print("1 = UP, 2 = Down, 3 = Left, 4 = Right, 5 = Park")
    print(first2pairs)

    # save results
    file_name = 'qtables/simpleq_'+ ffp.getName() +'_' + str(EPISODES) +'_' + str(datetime.today().strftime("%d-%m-%y %H %M %S")) +'.csv'
    np.save(file_name,Q)    



# %%
