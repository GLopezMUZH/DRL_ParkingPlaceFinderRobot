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
# %%
from datetime import datetime
import time
import numpy as np
import collections
import matplotlib.pylab as plt
import networkx as nx
from PIL import Image
import cv2
import scipy.misc
import csv
#from tqdm import tqdm
from tqdm.autonotebook import tqdm

import os
os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import Parking_lot
from Parking_lot import Parking_Lot
from Park_Finder_Agent import Park_Finder_Agent
from Park_Finder_Agent import Reward_Parameters

# %%
"""
#TODO:
    - Enable rendering with agent on the terminal state for better understanding
"""
# %%

class Utils():
    def print_frames(self, frames):
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

    def print_frame(self, frame):
        print("--- Frame ----------------")
        print(f"Timestep: {frame['step']+1}")
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

    def print_frame_summary(self,frame):
        print("--- Frame summary -----------")
        print(f"Timestep: {frame['step']+1}")
        print(f"Resulting state: {frame['resulting state']}")
        print(f"Path lenght: {len(frame['state history'])}")
        print(f"Reward: {frame['reward']}")
        print(f"Found parking: {frame['new start']}")
        print(f"Walking distance: {frame['walk distance']}")
        print(f"Driving distance: {frame['drive distance']}")

    def maxAction(self, Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = 0
        # if all the values are 0 in the Q table, pick random an action (otherwise it would always chose the first one)
        if sum(values) == 0:
            # print("picking random")
            action = np.random.randint(0, 4)
        else:
            # if there is already something learned, pick the one which has the highest reward attached to it
            action = np.argmax(values)
        return actions[action]


    def make_combo_plot(self, a, b, save_file=False, file_name_combo_plot='None', title_str='', subtitle_str=''):
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
        if title_str:
            plt.title(title_str, y=1.15, fontsize=14)
        if subtitle_str:
            plt.suptitle(subtitle_str, y=1.0)

        fig.tight_layout()
        if save_file:
            plt.savefig(file_name_combo_plot)
        plt.show()

    def save_frames(self, file_sufix, nr_episodes, frames):
        # save frames as csv but extension txt because delimier is ; and cells include ,'s
        file_name_frames_csv = 'qtables/simpleq_frames_' + file_sufix + '_' + str(nr_episodes) + \
            '_' + str(datetime.today().strftime("%d-%m-%y %H %M %S")) + '.txt'
        with open(file_name_frames_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(['Timestep', 'State', 'Action', 'Resulting state', 'Reward', 'Found parking',
                            'Path', 'Action history', 'Reward history', 'Walking distance', 'Driving distance','Step'])
            for i, frame in enumerate(frames):
                writer.writerow([(i + 1), frame['state'], frame['action'], frame['resulting state'], frame['reward'], frame['new start'],
                                frame['state history'], frame['action history'], frame['reward history'], frame['walk distance'], frame['drive distance'],frame['step']])

    def save_q_table(self, file_sufix, nr_episodes, Q):
        # save q-table as csv
        file_name_q_csv = 'qtables/simpleq_' + file_sufix + '_' + str(nr_episodes) + \
            '_' + str(datetime.today().strftime("%y-%m-%d %H %M %S")) + '.csv'
        with open(file_name_q_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for key, value in Q.items():
                writer.writerow([key[0], key[1], value])
        
        # save q-table as numpy
        file_name_q_np = 'qtables/simpleq_' + file_sufix + '_' + str(nr_episodes) + \
            '_' + str(datetime.today().strftime("%y-%m-%d %H %M %S"))
        np.save(file_name_q_np, Q)


# %%
# model hyperparameters
class Learning_Model_Parameters():
    def __init__(self,episodes, alpha=0.1, gamma=1.0):
        self.EPISODES = episodes # 55000
        self.ALPHA = alpha
        self.GAMMA = gamma

#%%
def create_parking_scenario(nr_episodes, uniform_distribution_p_value,
                            nr_parking_slots_per_lane, nr_parking_lanes,
                            parking_lane_depth, debug, draw_graph, save_plot_graph):
    ffp = Parking_lot.Filling_Function_Parameters(uniform_distribution_p_value=uniform_distribution_p_value)
    ldp = Parking_lot.Lane_Direction_Parameters()
    reward_parameters = Reward_Parameters()
    lmp = Learning_Model_Parameters(episodes=nr_episodes)

    file_name_parking_lot_plot = 'qtables/parking_lot_' + ffp.getName() + '_' + str(nr_episodes) + '_' + str(datetime.today().strftime("%y-%m-%d %H %M %S")) + '.png'

    parking_environment = Parking_Lot(lane_direction_paramenters=ldp,
                                        filling_function_parameters=ffp,
                                        nr_parking_slots_per_lane=nr_parking_slots_per_lane,
                                        nr_parking_lanes=nr_parking_lanes,
                                        parking_lane_depth=parking_lane_depth,
                                        debug=debug,
                                        draw_graph=draw_graph,
                                        save_plot_graph=save_plot_graph, 
                                        file_name_parking_lot_plot=file_name_parking_lot_plot,
                                        show_summary=False
                                        )

    agent = Park_Finder_Agent(
        reward_parameters=reward_parameters, parking_environment=parking_environment)
        
    return agent, lmp, parking_environment, ffp.getName()


# %%
def doLearning(agent: Park_Finder_Agent, parking_environment: Parking_Lot,
               Q: dict, lmp: Learning_Model_Parameters,
               debug=False, show=False, show_frames=False,
               save_qt=True, save_frames=False, plot_rewards = False, 
               file_sufix=''):

    def __printDebug(*val):
        if debug:
            print(list(val))

    # started with empty Q table, initialize to zeros
    if not Q:
        for state in agent.stateSpacePlus:
            for action in agent.possibleActions:
                Q[state, action] = 0



    utils = Utils()

    print('Start learning: '+str(datetime.today().strftime("%d-%m-%y %H %M %S"))+' - nr episodes: '+str(lmp.EPISODES))

    p_lot = parking_environment.get_env()
    ffp = parking_environment.filling_function_parameters

    frames = []  # for information
    fig = plt.figure()

    eps = 1.0
    num_episodes = lmp.EPISODES
    learning_rewards = np.zeros(num_episodes)
    epsilon = np.zeros(num_episodes)
    UPPER_LIMIT = lmp.EPISODES/100
    last_rewards = collections.deque(maxlen=int(UPPER_LIMIT))

    print('nr_occupied_parking_slots: '+str(parking_environment.nr_occupied_parking_slots))
    print('best parking slot: '+str(parking_environment.best_parking_slot))

    best_walk_distance = nx.shortest_path_length(
                    p_lot, source=parking_environment.best_parking_slot, target=max(agent.parking_lot.nodes))

    print('best_walk_distance: ' + str(best_walk_distance))
    walk_distance = 0

    pbar = tqdm(range(num_episodes))
    for i in pbar:
        # observation = current state
        observation = agent.agentPosition
        # Every xth episode we reset the car to position 0 and start again
        if i % (UPPER_LIMIT) == 0:
            observation = agent.reset()
        done = False

        ep_rewards = 0
        history = []
        action_history = []
        reward_history = []

        while not done:
            finish = False
            rand = np.random.random()
            action = utils.maxAction(Q, observation, agent.possibleActions) if rand < (
                1-eps) else agent.actionSpaceSample()
            # ovservation_ is a trial step
            observation_, reward, done, info = agent.step(action)

            # making sure that when parking lot was reached or a crash with a parked car occures, 
            # we terminate this episode (just experimental, should be handled in the getReward function)
            resulting_state = observation+agent.actionSpace[action]

            if done:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if (cv2.waitKey(50) & 0xFF) == ord('q'):
                    break
            else:
                if (cv2.waitKey(1) & 0x7F) == ord('q'):
                    break

            # agent will park
            if observation_+agent.actionSpace[action] in agent.vacant_list and action == 5:
                finish = True

            ep_rewards += reward

            action_ = utils.maxAction(Q, observation_, agent.possibleActions)
            Q[observation, action] = Q[observation, action] + lmp.ALPHA * \
                (reward + lmp.GAMMA*Q[observation_,
                                      action_] - Q[observation, action])
            history.append(observation)
            action_history.append(action)
            reward_history.append(ep_rewards)
            observation = observation_

            if resulting_state in agent.vacant_list and action == 5:
                walk_distance = nx.shortest_path_length(
                    p_lot, source=resulting_state, target=max(agent.parking_lot.nodes))
                drive_distance = nx.shortest_path_length(
                    p_lot, source=0, target=resulting_state)
            
            if finish:
                __printDebug("start a new episode")
                done = True
                agent.reset()
                history.append(resulting_state)
                frames.append({'state': observation, 'resulting state': resulting_state, 'state history': history[:-1],
                               'action history': action_history, 'reward history': reward_history,
                               'action': action, 'reward': ep_rewards, 'new start': done, 'walk distance': walk_distance,
                               'drive distance': drive_distance, 'step': i})
            else:
                walk_distance = 0
                drive_distance = 0

            if show:
                img = scipy.misc.toimage(agent.grid)
                # resizing so we can see our agent in all its glory.
                img = img.resize((500, 500))
                cv2.imshow("Parking Agent", np.array(img))

        if eps - 2 / num_episodes > 0:
            eps -= 2 / num_episodes
        else:
            eps = 0

        if (debug):
            pbar.set_description("Reward: {}, Parking at: {} Epsilon: {}".format(
                round(ep_rewards, 2), resulting_state, round(eps, 2)))

        epsilon[i] = eps
        if ep_rewards:
            learning_rewards[i] = ep_rewards
            last_rewards.append(ep_rewards)


        if len(last_rewards) == UPPER_LIMIT:  
            if  len(list(set(last_rewards))) == 1 and agent.reward_parameters.PARKING_REWARD-drive_distance == last_rewards[-1]:
                print("Early stopping because of no changes")
                epsilon = epsilon[epsilon != 0]
                learning_rewards = learning_rewards[learning_rewards != 0]
                break

        # totalRewards[i] = epRewards

    print('End learning: '+str(datetime.today().strftime("%d-%m-%y %H %M %S")))

    if not file_sufix:
        file_sufix = ffp.getName()


    if show_frames:
        print(utils.print_frames(frames))

    if save_frames:
        utils.save_frames(file_sufix, lmp.EPISODES, frames)

    if save_qt:
        # save q-table as numpy
        utils.save_q_table(file_sufix, lmp.EPISODES, Q)

    if plot_rewards:
        # plt.plot(totalRewards)
        file_name_combo_plot = 'qtables/plot_rewards_eps_' + ffp.getName() + '_' + \
            str(lmp.EPISODES) + '_' + \
            str(datetime.today().strftime("%y-%m-%d %H %M %S")) + '.png'
        title_str = ffp.getName() + '_' + str(lmp.EPISODES)
        subtitle_str = ' Alpha: ' + str('{0:.4f}'.format(lmp.ALPHA)) + ', Gamma: ' + str('{0:.4f}'.format(lmp.GAMMA))
        utils.make_combo_plot(learning_rewards, epsilon, save_file=True,
                            file_name_combo_plot=file_name_combo_plot, 
                            title_str=title_str, subtitle_str=subtitle_str)

    return Q, frames[-1]


#%%
# TODO
# delete Learning_Model_Parameters we are not learning
def doParking(agent: Park_Finder_Agent, parking_environment: Parking_Lot,
               Q: dict, max_episode_attempts=100,
               debug=False, show=False,
               save_frames=False):

    def __printDebug(*val):
        if debug:
            print(list(val))

    # started with empty Q table, initialize to zeros
    if not Q:
        return -1 #TODO raise exception

    utils = Utils()
    results = Parking_Results()

    print('Start parking: '+str(datetime.today().strftime("%d-%m-%y %H %M %S"))+' - nr episodes: '+str(max_episode_attempts))

    p_lot = parking_environment.get_env()
    ffp = parking_environment.filling_function_parameters

    frames = []  # for information
    fig = plt.figure()

    eps = 1.0
    num_episodes = max_episode_attempts
    total_rewards = np.zeros(num_episodes)
    epsilon = np.zeros(num_episodes)
    UPPER_LIMIT = max_episode_attempts/100
    last_rewards = collections.deque(maxlen=int(UPPER_LIMIT))
    number_of_resets = 0

    print('nr_occupied_parking_slots: '+str(parking_environment.nr_occupied_parking_slots))
    print('best parking slot: '+str(parking_environment.best_parking_slot))

    best_walk_distance = nx.shortest_path_length(
                    p_lot, source=parking_environment.best_parking_slot, target=max(agent.parking_lot.nodes))
                    
    print('best_walk_distance: ' + str(best_walk_distance))

    pbar = tqdm(range(num_episodes))
    for i in pbar:
        # observation = current state
        observation = agent.agentPosition
        # Every xth episode we reset the car to position 0 and start again
        if i % (UPPER_LIMIT) == 0:
            number_of_resets += 1
            observation = agent.reset()
        done = False

        ep_rewards = 0
        history = []
        action_history = []
        reward_history = []

        while not done:
            finish = False
            rand = np.random.random()
            action = utils.maxAction(Q, observation, agent.possibleActions) if rand < (
                1-eps) else agent.actionSpaceSample()
            # ovservation_ is a trial step
            observation_, reward, done, info = agent.step(action)

            # making sure that when parking lot was reached or a crash with a parked car occures,
            # we terminate this episode (just experimental, should be handled in the getReward function)
            resulting_state = observation+agent.actionSpace[action]

            if done:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if (cv2.waitKey(50) & 0xFF) == ord('q'):
                    break
            else:
                if (cv2.waitKey(1) & 0x7F) == ord('q'):
                    break

            # agent will park
            if observation_+agent.actionSpace[action] in agent.vacant_list and action == 5:
                finish = True

            ep_rewards += reward

            history.append(observation)
            action_history.append(action)
            reward_history.append(ep_rewards)
            observation = observation_

            if resulting_state in agent.vacant_list and action == 5:
                walk_distance = nx.shortest_path_length(
                    p_lot, source=resulting_state, target=max(agent.parking_lot.nodes))
                drive_distance = nx.shortest_path_length(
                    p_lot, source=0, target=resulting_state)
            
            if finish:
                __printDebug("start a new episode")
                done = True
                agent.reset()
                history.append(resulting_state)
                frames.append({'state': observation, 'resulting state': resulting_state, 'state history': history[:-1],
                               'action history': action_history, 'reward history': reward_history,
                               'action': action, 'reward': ep_rewards, 'new start': done, 'walk distance': walk_distance,
                               'drive distance': drive_distance})
            else:
                walk_distance = 0
                drive_distance = 0

            if show:
                img = scipy.misc.toimage(agent.grid)
                # resizing so we can see our agent in all its glory.
                img = img.resize((500, 500))
                cv2.imshow("Parking Agent", np.array(img))

        if eps - 2 / num_episodes > 0:
            eps -= 2 / num_episodes
        else:
            eps = 0

        epsilon[i] = eps
        if ep_rewards:
            learning_rewards[i] = ep_rewards
            last_rewards.append(ep_rewards)

        if len(last_rewards) == UPPER_LIMIT and len(list(set(last_rewards))) == 1 and agent.reward_parameters.PARKING_REWARD-drive_distance == last_rewards[-1]:
            print("Early stopping because of no changes. Run episodes: "+str(i))
            epsilon = epsilon[epsilon != 0]
            learning_rewards = learning_rewards[learning_rewards != 0]
            break

        # agent has parked
        if  done == True:
            results.walk_distance=walk_distance
            results.found_slot=resulting_state
            results.frames=frames
            break

        # totalRewards[i] = epRewards

    print('End parking: '+str(datetime.today().strftime("%d-%m-%y %H %M %S")))

    if save_frames:
        utils.save_frames(ffp, max_episode_attempts, frames)

    # plt.plot(totalRewards)
    file_name_combo_plot = 'qtables/plot_rewards_eps_' + ffp.getName() + '_' + \
        str(max_episode_attempts) + '_' + \
        str(datetime.today().strftime("%y-%m-%d %H %M %S")) + '.png'

    utils.make_combo_plot(learning_rewards, epsilon, save_file=True,
                    file_name_combo_plot=file_name_combo_plot)

    return results


#%%
def create_base_test_scenario(nr_episodes):
    return create_parking_scenario(nr_episodes=nr_episodes, 
                                    uniform_distribution_p_value=0.5,
                                    nr_parking_slots_per_lane=5,
                                    nr_parking_lanes=4,
                                    parking_lane_depth=2,
                                    debug=False,
                                    draw_graph=False,
                                    save_plot_graph=True
                                    )


# %%
# EXAMPLE - simple
# 
#if __name__ == '__main__':
def example_1():
    utils = Utils()
    agent, lmp, parking_environment, file_sufix = create_base_test_scenario(nr_episodes=100)
    Q = {}
    Q_1, last_frame = doLearning(agent=agent, parking_environment=parking_environment, 
                                    Q=Q, lmp=lmp, save_qt=False, 
                                    save_frames=False,plot_rewards=True)

    utils.print_frame_summary(last_frame)
    
    return Q_1

# Q_1 = example_1()

#%%
# Example 2 - use previous Q_table and overwrite it
# reset parking lot, plot it again and do learning over new parking configuration with input from previous q-t
def example_2():
    agent, lmp, parking_environment, file_sufix = create_base_test_scenario(nr_episodes=100)
    Q = {}

    Q_1, last_frame = doLearning(agent=agent, parking_environment=parking_environment, 
                                Q=Q, lmp=lmp, save_qt=True,
                                save_frames=True, plot_rewards=True)

    parking_environment.clear_occupation()
    parking_environment.fill_parking_slots()
    agent.reset()
    file_name_parking_lot_plot = 'qtables/parking_lot_' + file_sufix + '_' + \
        str(lmp.EPISODES) + '_' + \
        str(datetime.today().strftime("%y-%m-%d %H %M %S")) + '.png'

    parking_environment.plot(save_file=True,file_name_parking_lot_plot=file_name_parking_lot_plot)

    Q_2, last_frame = doLearning(agent=agent, parking_environment=parking_environment, 
                        Q=Q_1, lmp=lmp, save_qt=True, save_frames=True,
                        plot_rewards=True)

# example_2()

#%%
def plot_field_from_last_frames(frames, field_name, x_range, x_label='', title_str='', subtitle_str=''):
    plot_array = []
    for i in range(len(frames)):
        plot_array.append(frames[i][field_name])

    fig, ax1 = plt.subplots()
    plt.plot(x_range, plot_array)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(field_name)

    if title_str:
        plt.title(title_str, y=1.15, fontsize=14)
    if subtitle_str:
        plt.suptitle(subtitle_str, y=1.0)

    fig.tight_layout()
    plt.savefig('qtables/'+field_name+'_on_'+x_label+'.png')
    plt.show()


# %%
# example 3, compare learning with different parameters
def example_changes_on_alpha():
    utils = Utils()
    agent, lmp, parking_environment, file_sufix = create_base_test_scenario(
        nr_episodes=50)

    values_alpha = np.linspace(0.01, 0.9, 2)
    values_alpha = np.append(values_alpha, [0.1])
    values_alpha.sort()

    last_frames = []

    for v_alpha in values_alpha:
        Q = {}
        Q_1 = {}
        for state in agent.stateSpacePlus:
            for action in agent.possibleActions:
                Q[state, action] = 0

        lmp.ALPHA = v_alpha
        Q_1, last_frame = doLearning(agent=agent, parking_environment=parking_environment,
                                     Q=Q, lmp=lmp, save_qt=False, save_frames=False,
                                     plot_rewards=True)

        last_frames.append(last_frame)
        utils.print_frame_summary(last_frame)

    return last_frames, values_alpha

#last_frames, values_alpha = example_changes_on_alpha()
#plot_field_from_last_frames(frames=last_frames, field_name='reward',
#                                x_range=values_alpha, x_label='alpha', title_str='Changes in Alpha')
